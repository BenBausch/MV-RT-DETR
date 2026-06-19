"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import csv
import copy
import datetime
from pathlib import Path

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
        train_ds = getattr(self.train_dataloader, 'dataset', None)
        train_size = len(train_ds) if train_ds is not None else None
        wandb_logger.setup(self.cfg, class_names=class_names, train_size=train_size)

        # per-tooth eval operating point (score / IoU threshold), shared with the
        # end-of-training test pass. Defaults: score 0.5 / IoU 0.5.
        pt_cfg = self.cfg.yaml_cfg.get('per_tooth_eval', {}) or {}
        pt_score_thr = float(pt_cfg.get('score_thr', 0.5))
        pt_iou_thr = float(pt_cfg.get('iou_thr', 0.5))

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
                pt_score_thr=pt_score_thr,
                pt_iou_thr=pt_iou_thr,
            )

            # per-tooth metrics are a nested dict, not a list of floats -> handle
            # separately from the generic (per_tooth_f1) list-of-floats stats.
            pt_stats = test_stats.pop('per_tooth', None)
            if pt_stats is not None and self.output_dir and dist_utils.is_main_process():
                write_confusion(pt_stats.get('confusion'), pt_stats.get('names', {}),
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

            # ---- wandb per-epoch validation logging (per-tooth, incl. per-class) ----
            if wandb_logger.is_active():
                if pt_stats is not None:
                    wandb_logger.log_per_tooth(pt_stats, epoch, prefix='val')
                wandb_logger.log_summary({'best/micro_f1': best_stat.get('per_tooth_f1', 0),
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

        # ---- end-of-training TEST evaluation on the BEST checkpoint ----
        # Runs exactly once, after all epochs; uses the test split (not val) and the
        # best.pth selected by val micro-F1 (NOT the last-epoch weights).
        self._test_eval(class_names, best_stat.get('epoch', -1), pt_score_thr, pt_iou_thr)

        wandb_logger.finish()


    def _build_test_dataloader(self):
        """Derive a test-split loader from the (already-merged) val_dataloader config,
        so transforms (incl. OrientationChannel) match val exactly; only the split file
        differs. Returns None if no test split file is found."""
        if 'val_dataloader' not in self.cfg.yaml_cfg:
            return None
        pt_cfg = self.cfg.yaml_cfg.get('per_tooth_eval', {}) or {}
        vcfg = copy.deepcopy(self.cfg.yaml_cfg['val_dataloader'])
        val_split = vcfg.get('dataset', {}).get('split_file')
        if not val_split:
            return None
        test_split = pt_cfg.get('test_split_file') or str(Path(val_split).with_name('test.txt'))
        if not Path(test_split).exists():
            print(f'[test-eval] no test split at {test_split}; skipping test evaluation')
            return None
        vcfg['dataset']['split_file'] = test_split
        self.cfg.yaml_cfg['test_dataloader'] = vcfg
        loader = self.cfg.build_dataloader('test_dataloader')
        return dist_utils.warp_loader(loader, shuffle=False)

    def _test_eval(self, class_names, best_epoch, pt_score_thr, pt_iou_thr):
        if not dist_utils.is_main_process():
            return
        test_loader = self._build_test_dataloader()
        if test_loader is None:
            return

        # reload the best checkpoint (fall back to last.pth) so test reflects the
        # selected model, not the final-epoch weights.
        ckpt = None
        if self.output_dir:
            for name in ('best.pth', 'last.pth'):
                if (self.output_dir / name).exists():
                    ckpt = self.output_dir / name
                    break
        if ckpt is not None:
            print(f'[test-eval] loading checkpoint {ckpt}')
            self.load_resume_state(str(ckpt))
        module = self.ema.module if self.ema else self.model

        print(f'[test-eval] running per-tooth metrics on the TEST split '
              f'(score_thr={pt_score_thr}, IoU={pt_iou_thr})')
        test_stats, _ = evaluate(
            module, self.criterion, self.postprocessor, test_loader,
            None, self.device, ul_names=class_names,
            pt_score_thr=pt_score_thr, pt_iou_thr=pt_iou_thr,
        )
        pt = test_stats.get('per_tooth')
        if pt is None:
            return

        if self.output_dir:
            test_dir = self.output_dir / 'test'
            write_confusion(pt.get('confusion'), pt.get('names', {}), test_dir)
            self._write_per_class_csv(pt, test_dir / 'test_per_class.csv', pt_score_thr, pt_iou_thr)

        if wandb_logger.is_active():
            wandb_logger.log_per_tooth(pt, epoch=None, prefix='test')
            wandb_logger.log_summary({'test/source_epoch': best_epoch})

    @staticmethod
    def _write_per_class_csv(pt, path, score_thr, iou_thr):
        names = pt.get('names', {})
        rows = []
        for c, d in sorted(pt['per_class'].items()):
            rows.append(dict(
                fdi=names.get(c, str(c)), class_id=c, score_thr=score_thr, iou=iou_thr,
                support=d['support'], tp=d['tp'], fp=d['fp'], fn=d['fn'], dup=d['dup'],
                correct=d['correct'], misloc=d['misloc'], miscls=d['miscls'], missed=d['missed'],
                precision=round(d['p'], 4), recall=round(d['r'], 4), f1=round(d['f1'], 4),
            ))
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'[test-eval] wrote per-class metrics -> {path}')


    def val(self, ):
        self.eval()

        pt_cfg = self.cfg.yaml_cfg.get('per_tooth_eval', {}) or {}
        pt_score_thr = float(pt_cfg.get('score_thr', 0.5))
        pt_iou_thr = float(pt_cfg.get('iou_thr', 0.5))

        module = self.ema.module if self.ema else self.model
        val_ds = getattr(self.val_dataloader, 'dataset', None)
        class_names = getattr(val_ds, 'category2name', None)
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device, ul_names=class_names,
                pt_score_thr=pt_score_thr, pt_iou_thr=pt_iou_thr)

        pt_stats = test_stats.get('per_tooth')
        if pt_stats is not None and self.output_dir and dist_utils.is_main_process():
            write_confusion(pt_stats.get('confusion'), pt_stats.get('names', {}),
                            self.output_dir)

        # COCO is disabled for ortho runs (no evaluator); only dump when present.
        if self.output_dir and coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
