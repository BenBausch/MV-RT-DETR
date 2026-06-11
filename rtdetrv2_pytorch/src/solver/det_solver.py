"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils, wandb_logger
from ..misc.ultralytics_eval import write_confusion

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        # ---- wandb (opt-in via use_wandb in config) ----
        # NOTE: metrics only. Images are NEVER uploaded (patient data is confidential).
        val_ds = getattr(self.val_dataloader, 'dataset', None)
        class_names = getattr(val_ds, 'category2name', None)
        wandb_logger.setup(self.cfg, class_names=class_names)

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                ul_names=class_names,
            )

            # ultralytics metrics are a nested dict, not a list of floats -> handle
            # separately from the generic (coco_eval_bbox) list-of-floats stats.
            ul_stats = test_stats.pop('ultralytics', None)
            if ul_stats is not None and self.output_dir and dist_utils.is_main_process():
                write_confusion(ul_stats.get('confusion'), ul_stats.get('names', {}),
                                self.output_dir)

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            # ---- wandb per-epoch logging ----
            if wandb_logger.is_active():
                wandb_logger.log_coco_stats(test_stats.get('coco_eval_bbox'), epoch)
                if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
                    wandb_logger.log_per_class_ap(coco_evaluator.coco_eval['bbox'], epoch)
                if ul_stats is not None:
                    wandb_logger.log_ultralytics(ul_stats, epoch)
                wandb_logger.log_summary({'best/AP': best_stat.get('coco_eval_bbox', 0),
                                          'best/epoch': best_stat.get('epoch', -1)})

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        wandb_logger.finish()


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        val_ds = getattr(self.val_dataloader, 'dataset', None)
        class_names = getattr(val_ds, 'category2name', None)
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device, ul_names=class_names)

        ul_stats = test_stats.get('ultralytics')
        if ul_stats is not None and self.output_dir and dist_utils.is_main_process():
            write_confusion(ul_stats.get('confusion'), ul_stats.get('names', {}),
                            self.output_dir)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
