import os
from typing import Dict, Tuple
import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np

import transformers
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer import TRAINER_STATE_NAME

from src.model import get_padding_mask


class BaseTrainer(Trainer):
    def __init__(self, **kwargs):
        self.k = kwargs.pop("k", 100)  # for retrieval metrics
        self.save_weights_only = kwargs.pop("save_weights_only", True)
        kwargs.pop("track_id_to_artist_id", None)  # will be set explicitly after initialization
        self.loss_fn_train = torch.nn.CrossEntropyLoss(
            reduction='none',
            label_smoothing=kwargs.pop("label_smoothing", 0.0)
        )
        self.loss_fn_valid = torch.nn.CrossEntropyLoss(reduction='none')

        super().__init__(**kwargs)

        self.track_id_to_artist_id = None
        self.logger = transformers.logging.get_logger("trainer")

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> torch.utils.data.DataLoader:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        return DataLoader(
            ds,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _postprocess_metrics(self, metrics: Dict) -> Dict:
        # * add prefix
        # * make json-serializable
        metrics = {f"eval_{k}": float(v) for k, v in metrics.items()}

        # из родительского класса
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.save_weights_only:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self.args.output_dir
            self.store_flos()
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                        self.state.best_metric is None
                        or self.state.best_model_checkpoint is None
                        or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
        else:
            super()._save_checkpoint(model=model, trial=trial, metrics=metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        В самом конце обучения может вылазить ZeroDivisionError при расчёте
        лосса для записи в лог. Внёс соответствующий фикс
        """
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # в конце обучения коллективная коммуникация в этом месте вызывает зависание
            # TODO: разобраться
            tr_loss_scalar = tr_loss.item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            d = max(self.state.global_step - self._globalstep_last_logged, 1)  # my fix here
            logs["loss"] = round(tr_loss_scalar / d, 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.

        добавил логирование, чтоб метрики попадали в файл с логами
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        if self.args.local_rank in [-1, 0]:
            # tensorboard.compat.tensorflow_stub.errors.AlreadyExistsError: Directory already exists
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
            self.logger.info(output)

    def store_flos(self):
        """
        наблюдал зависания из-за этой функции
        TODO: разобраться, в чём дело
        """
        pass


class TrainerV1(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        track_ids = inputs["track_ids"]  # [N, T, M]
        artist_ids = inputs["artist_ids"]  # [N, T, M]

        track_ids_x = track_ids[:, :, 0]  # [N, T]
        queries = model(
            track_ids=track_ids_x,
            artist_ids=(artist_ids[:, :, 0] if artist_ids is not None else None)
        )  # [N, T, D]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        keys = model.get_items_embeddings(
            track_ids=track_ids[:, :, 1:],
            artist_ids=(artist_ids[:, :, 1:] if artist_ids is not None else None)
        )  # [N, T, M, D]

        # [N, T, 1, D] * [N, T, D, M] = [N, T, 1, M] -> squeeze(2) -> [N, T, M]
        logits = torch.matmul(queries.unsqueeze(2), keys.transpose(2, 3)).squeeze(2)  # [N, T, M]
        labels = torch.zeros((track_ids.shape[0], track_ids.shape[1])).long().to(self.args.device)  # [N, T]

        # compute per-example loss
        # * class dimention must be the second (in contrast to SparseCategoricalCrossEntropy in tf)
        # * output has the same shape as labels
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        per_example_loss = self.loss_fn_train(logits.transpose(1, 2), labels)  # [N, T]

        # mask loss
        padding_mask = get_padding_mask(track_ids_x, padding_idx=model.padding_idx)
        padding_mask = padding_mask.float()
        per_example_loss *= padding_mask  # [N, T]
        loss = per_example_loss.sum() / (padding_mask.sum() + 1e-6)  # []
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        loader = self.get_eval_dataloader(eval_dataset=ds)
        self.model.eval()

        n = self.model.num_tracks + self.model.num_special_tokens
        track_ids = torch.arange(n, device=self.args.device)
        keys = self.model.get_items_embeddings(track_ids=track_ids, artist_ids=self.track_id_to_artist_id)  # [V, D]
        keys = keys.T  # [D, V]

        num_examples = 0
        total_loss = 0.0
        mrr = 0.0
        accuracy = 0.0
        pbar = tqdm.tqdm(
            desc=f"rank {self.args.local_rank}",
            total=len(ds),
            position=self.args.local_rank,
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                track_ids = batch["track_ids"].to(self.args.device)
                artist_ids = batch["artist_ids"].to(self.args.device)
                batch_size = track_ids.shape[0]
                outputs = self.model(track_ids=track_ids[:, :, 0], artist_ids=artist_ids[:, :, 0])  # [N, T, D]
                padding_mask = get_padding_mask(track_ids[:, :, 0], padding_idx=self.model.padding_idx)  # [N, T]
                padding_mask = padding_mask.to(self.args.device)
                xs = torch.arange(batch_size).to(self.args.device)
                last_item_idx = padding_mask.long().sum(1) - 1  # [N]
                logits = torch.matmul(outputs[xs, last_item_idx, :], keys)  # [N, D] * [D, V] = [N, V]
                labels = track_ids[xs, last_item_idx, 1]  # [N]
                per_example_loss = self.loss_fn_valid(logits, labels)  # [N]
                total_loss += per_example_loss.sum().item()
                num_examples += batch_size
                indices = (logits * -1.0).argsort(-1)  # [N, V]
                accuracy += torch.eq(labels, indices[:, 0]).float().sum().item()

                # mrr
                for i in range(batch_size):
                    for j in range(self.k):
                        if indices[i, j] == labels[i]:
                            mrr += 1.0 / (j + 1.0)
                            break
                pbar.update(batch_size)
        pbar.close()

        # torch.cuda.empty_cache()

        metrics = {
            "loss": total_loss / num_examples,
            "mrr": mrr / num_examples,
            "accuracy": accuracy / num_examples
        }

        metrics = self._postprocess_metrics(metrics)
        return metrics


class TrainerV2(BaseTrainer):
    def __init__(self, **kwargs):
        self.m = kwargs.pop("num_candidates", 1000)
        super().__init__(**kwargs)

        self.V = self.model.num_tracks + self.model.num_special_tokens

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        track_ids_x = inputs["inputs"]  # [N, T]
        track_ids_y = inputs["targets"]  # [N, T]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            use_artist_emb = model.module.use_artist_emb
        else:
            use_artist_emb = model.use_artist_emb
        if use_artist_emb:
            artist_ids_x = self.track_id_to_artist_id[track_ids_x]  # [N, T]
        else:
            artist_ids_x = None
        queries = model(
            track_ids=track_ids_x,
            artist_ids=artist_ids_x
        )  # [N, T, D]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        track_ids_neg = self.generate_negatives_2d(track_ids_x)  # [N, T, M]
        track_ids_key = torch.cat([track_ids_y[:, :, None], track_ids_neg], dim=-1)  # [N, T, M + 1]

        if model.use_artist_emb:
            artist_ids_key = self.track_id_to_artist_id[track_ids_key]
        else:
            artist_ids_key = None

        keys = model.get_items_embeddings(
            track_ids=track_ids_key,
            artist_ids=artist_ids_key
        )  # [N, T, M, D]

        # [N, T, 1, D] * [N, T, D, M] = [N, T, 1, M] -> squeeze(2) -> [N, T, M]
        logits = torch.matmul(queries.unsqueeze(2), keys.transpose(2, 3)).squeeze(2)  # [N, T, M]
        labels = torch.zeros((track_ids_x.shape[0], track_ids_x.shape[1])).long().to(self.args.device)  # [N, T]

        # compute per-example loss
        # * class dimention must be the second (in contrast to SparseCategoricalCrossEntropy in tf)
        # * output has the same shape as labels
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        per_example_loss = self.loss_fn_train(logits.transpose(1, 2), labels)  # [N, T]

        # mask loss
        padding_mask = get_padding_mask(track_ids_x, padding_idx=model.padding_idx)
        fn_mask = torch.logical_not(torch.eq(track_ids_neg, track_ids_y[:, :, None]).any(-1))  # [N, T]
        loss_mask = torch.logical_and(padding_mask, fn_mask).float()
        per_example_loss *= loss_mask  # [N, T]
        loss = per_example_loss.sum() / (loss_mask.sum() + 1e-6)  # []
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        loader = self.get_eval_dataloader(eval_dataset=ds)
        self.model.eval()

        n = self.model.num_tracks + self.model.num_special_tokens
        track_ids = torch.arange(n, device=self.args.device)
        if self.model.use_artist_emb:
            artist_ids = self.track_id_to_artist_id
        else:
            artist_ids = None
        keys = self.model.get_items_embeddings(track_ids=track_ids, artist_ids=artist_ids)  # [V, D]
        keys = keys.T  # [D, V]

        num_examples = 0
        total_loss = 0.0
        mrr = 0.0
        accuracy = 0.0
        pbar = tqdm.tqdm(
            desc=f"rank {self.args.local_rank}",
            total=len(ds),
            position=self.args.local_rank,
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                track_ids_x = batch["inputs"].to(self.args.device)  # [N, T]
                track_ids_y = batch["targets"].to(self.args.device)  # [N, T]
                batch_size = track_ids_x.shape[0]
                if self.model.use_artist_emb:
                    artist_ids_x = self.track_id_to_artist_id[track_ids_x]  # [N, T]
                else:
                    artist_ids_x = None
                outputs = self.model(track_ids=track_ids_x, artist_ids=artist_ids_x)  # [N, T, D]
                padding_mask = get_padding_mask(track_ids_x, padding_idx=self.model.padding_idx)  # [N, T]
                padding_mask = padding_mask.to(self.args.device)
                xs = torch.arange(batch_size).to(self.args.device)
                last_item_idx = padding_mask.long().sum(1) - 1  # [N]
                logits = torch.matmul(outputs[xs, last_item_idx, :], keys)  # [N, D] * [D, V] = [N, V]
                labels = track_ids_y[xs, last_item_idx]  # [N]
                per_example_loss = self.loss_fn_valid(logits, labels)  # [N]
                total_loss += per_example_loss.sum().item()
                num_examples += batch_size
                indices = (logits * -1.0).argsort(-1)  # [N, V]
                accuracy += torch.eq(labels, indices[:, 0]).float().sum().item()

                # mrr
                for i in range(batch_size):
                    for j in range(self.k):
                        if indices[i, j] == labels[i]:
                            mrr += 1.0 / (j + 1.0)
                            break
                pbar.update(batch_size)
        pbar.close()

        torch.cuda.empty_cache()

        metrics = {
            "loss": total_loss / num_examples,
            "mrr": mrr / num_examples,
            "accuracy": accuracy / num_examples
        }

        metrics = self._postprocess_metrics(metrics)
        return metrics

    def generate_negatives_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, T], int64
        :return res: [N, T, M], int64
        """
        N = x.shape[0]
        T = x.shape[1]
        assert T <= self.m, f'{T} > {self.m}'
        samples = torch.randint(
            self.model.num_special_tokens, self.V, (N, T, self.m), device=self.args.device)  # [N, T, T + m]
        mask = torch.tril(torch.ones((T, self.m), device=self.args.device))  # [T, T + m]
        history = torch.zeros((N, T, self.m), device=self.args.device)  # [N, T, T + m]
        history[:, :, :T] = torch.tile(x[:, None, :], [1, T, 1])  # [N, T, T] <- [N, T, T]
        res = history * mask + samples * (1 - mask)  # [N, T, T + m]
        res = res.long()
        return res


class TrainerV3(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        artist_ids_x = inputs["inputs"]  # [N, T]
        artist_ids_y = inputs["targets"]  # [N, T]

        queries = model(track_ids=None, artist_ids=artist_ids_x)  # [N, T, D]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        keys = model.artist_emb.weight  # [V, D]

        # [N, T, D] * [1, D, V] = [N, T, V]
        logits = torch.matmul(queries, keys.T[None])  # [N, T, V]

        # compute per-example loss
        # * class dimention must be the second (in contrast to SparseCategoricalCrossEntropy in tf)
        # * output has the same shape as labels
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        per_example_loss = self.loss_fn_train(logits.transpose(1, 2), artist_ids_y)  # [N, T]

        # mask loss
        padding_mask = get_padding_mask(artist_ids_x, padding_idx=model.padding_idx)
        padding_mask = padding_mask.float()
        per_example_loss *= padding_mask  # [N, T]
        loss = per_example_loss.sum() / (padding_mask.sum() + 1e-6)  # []
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        loader = self.get_eval_dataloader(eval_dataset=ds)
        self.model.eval()

        num_pairs = 0
        num_examples = 0
        total_loss = 0.0
        accuracy = 0.0
        accuracy_last = 0.0
        mrr_last = 0.0
        pbar = tqdm.tqdm(
            desc=f"rank {self.args.local_rank}",
            total=len(ds),
            position=self.args.local_rank,
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                artist_ids_x = batch["inputs"].to(self.args.device)
                artist_ids_y = batch["targets"].to(self.args.device)
                batch_size = artist_ids_x.shape[0]
                num_examples += batch_size
                outputs = self.model(track_ids=None, artist_ids=artist_ids_x)  # [N, T, D]
                logits = torch.matmul(outputs, self.model.artist_emb.weight.T[None])  # [N, T, V]
                per_example_loss = self.loss_fn_valid(logits.transpose(1, 2), artist_ids_y)  # [N, T]
                padding_mask = get_padding_mask(artist_ids_x, padding_idx=self.model.padding_idx)  # [N, T]
                padding_mask = padding_mask.float()
                padding_mask = padding_mask.to(self.args.device)
                per_example_loss *= padding_mask
                total_loss += per_example_loss.sum().item()
                num_pairs += padding_mask.sum().item()
                indices = logits.argmax(-1)  # [N, T]
                accuracy += (torch.eq(artist_ids_y, indices).float() * padding_mask).sum().item()

                xs = torch.arange(batch_size).to(self.args.device)
                sequence_lengths = padding_mask.sum(1).long()
                last_item_idx = sequence_lengths - 1  # [N]
                preds_last = (logits[xs, last_item_idx, :] * -1.0).argsort(-1)  # [N, V]
                for i in range(batch_size):
                    label = artist_ids_y[i, last_item_idx[i]]
                    for j in range(self.k):
                        if preds_last[i, j].item() == label:
                            mrr_last += 1.0 / (j + 1.0)
                            if j == 0:
                                accuracy_last += 1
                            break
                pbar.update(batch_size)
            pbar.close()

            torch.cuda.empty_cache()

            metrics = {
                "loss": total_loss / num_pairs,
                "accuracy": accuracy / num_pairs,
                "accuracy_last": accuracy_last / num_examples,
                "mrr_last": mrr_last / num_examples
            }
            metrics = self._postprocess_metrics(metrics)
        return metrics


class TrainerV4(BaseTrainer):
    """
    Too memory expensive
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        track_ids_x = inputs["track_ids_x"]  # [N, T]
        track_ids_y = inputs["track_ids_y"]  # [N, T]
        artist_ids_x = inputs["artist_ids_x"]  # [N, T]
        artist_ids_y = inputs["artist_ids_y"]  # [N, T]

        queries = model(track_ids=track_ids_x, artist_ids=artist_ids_x)  # [N, T, D]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        keys_track = model.track_emb.weight  # [V, D]
        keys_artist = model.artist_emb.weight

        # [N, T, D] * [1, D, V] = [N, T, V]
        logits_track = torch.matmul(queries, keys_track.T[None])  # [N, T, V_track]
        logits_artist = torch.matmul(queries, keys_artist.T[None])  # [N, T, V_artist]

        per_example_loss_track = self.loss_fn_train(logits_track.transpose(1, 2), track_ids_y)  # [N, T]
        per_example_loss_artist = self.loss_fn_train(logits_artist.transpose(1, 2), artist_ids_y)  # [N, T]

        # mask loss
        padding_mask = get_padding_mask(track_ids_x, padding_idx=model.padding_idx)
        padding_mask = padding_mask.float()

        loss_track = (per_example_loss_track * padding_mask).sum() / (padding_mask.sum() + 1e-6)  # []
        loss_artist = (per_example_loss_artist * padding_mask).sum() / (padding_mask.sum() + 1e-6)  # []
        loss = loss_track * 0.5 + loss_artist * 0.5

        if (self.args.local_rank == 0) & (self.state.global_step % self.args.logging_steps == 0):
            logs = {
                "loss": round(loss.item(), 4),
                "loss_track": round(loss_track.item(), 4),
                "loss_artist": round(loss_artist.item(), 4),
                "step": self.state.global_step,
            }
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)
            self.logger.info(logs)
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        loader = self.get_eval_dataloader(eval_dataset=ds)
        self.model.eval()

        num_examples = 0
        loss_track = 0.0
        loss_artist = 0.0
        accuracy_track = 0.0
        accuracy_artist = 0.0
        pbar = tqdm.tqdm(
            desc=f"rank {self.args.local_rank}",
            total=len(ds),
            position=self.args.local_rank,
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                track_ids_x = batch["track_ids_x"].to(self.args.device)  # [N, T]
                track_ids_y = batch["track_ids_y"].to(self.args.device)  # [N, T]
                artist_ids_x = batch["artist_ids_x"].to(self.args.device)  # [N, T]
                artist_ids_y = batch["artist_ids_y"].to(self.args.device)  # [N, T]
                batch_size = artist_ids_x.shape[0]
                outputs = self.model(track_ids=track_ids_x, artist_ids=artist_ids_x)  # [N, T, D]

                padding_mask = get_padding_mask(artist_ids_x, padding_idx=self.model.padding_idx)  # [N, T]
                padding_mask = padding_mask.float()
                padding_mask = padding_mask.to(self.args.device)
                num_examples += padding_mask.sum().item()

                loss_i, acc_i = self._get_loss_and_accuracy(outputs, track_ids_y, padding_mask)
                loss_track += loss_i
                accuracy_track += acc_i
                loss_i, acc_i = self._get_loss_and_accuracy(outputs, artist_ids_y, padding_mask)
                loss_artist += loss_i
                accuracy_artist += acc_i

                pbar.update(batch_size)
        pbar.close()

        torch.cuda.empty_cache()

        metrics = {
            "loss": (loss_track * 0.5 + loss_artist * 0.5) / num_examples,
            "loss_track": loss_track / num_examples,
            "loss_artist": loss_artist / num_examples,
            "accuracy_track": accuracy_track / num_examples,
            "accuracy_artist": accuracy_artist / num_examples
        }

        metrics = self._postprocess_metrics(metrics)
        return metrics

    def _get_loss_and_accuracy(self, outputs, targets, mask) -> Tuple[float, float]:
        logits = torch.matmul(outputs, self.model.track_emb.weight.T[None])  # [N, T, V]
        per_example_loss = self.loss_fn_valid(logits.transpose(1, 2), targets)  # [N, T]
        loss = (per_example_loss * mask).sum().item()
        preds = logits.argmax(-1)  # [N, T]
        accuracy = (torch.eq(targets, preds).float() * mask).sum().item()
        return loss, accuracy


class TrainerV5(TrainerV2):
    """
    Learns to predict track and artist
    """
    def __init__(self, **kwargs):
        self.w_track_loss = kwargs.pop("w_track", 0.5)
        self.w_artist_loss = kwargs.pop("w_artist", 0.5)
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        track_ids_x = inputs["inputs"]  # [N, T]
        track_ids_y = inputs["targets"]  # [N, T]

        artist_ids_x = self.track_id_to_artist_id[track_ids_x]  # [N, T]
        queries = model(
            track_ids=track_ids_x,
            artist_ids=artist_ids_x
        )  # [N, T, D]

        track_ids_neg = self.generate_negatives_2d(track_ids_x)  # [N, T, M]
        track_ids_key = torch.cat([track_ids_y[:, :, None], track_ids_neg], dim=-1)  # [N, T, M + 1]
        artist_ids_key = self.track_id_to_artist_id[track_ids_key]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        # track
        keys = model.get_items_embeddings(
            track_ids=track_ids_key,
            artist_ids=artist_ids_key
        )  # [N, T, M, D]
        logits = torch.matmul(queries.unsqueeze(2), keys.transpose(2, 3)).squeeze(2)  # [N, T, M]
        labels = torch.zeros((track_ids_x.shape[0], track_ids_x.shape[1])).long().to(self.args.device)  # [N, T]
        per_example_loss = self.loss_fn_train(logits.transpose(1, 2), labels)  # [N, T]
        padding_mask = get_padding_mask(track_ids_x, padding_idx=model.padding_idx)
        fn_mask = torch.logical_not(torch.eq(track_ids_neg, track_ids_y[:, :, None]).any(-1))  # [N, T]
        loss_mask = torch.logical_and(padding_mask, fn_mask).float()
        per_example_loss *= loss_mask  # [N, T]
        loss_track = per_example_loss.sum() / (loss_mask.sum() + 1e-6)  # []

        # artist
        keys = model.artist_emb.weight  # [V, D]
        logits = torch.matmul(queries, keys.T[None])  # [N, T, V]
        artist_ids_y = self.track_id_to_artist_id[track_ids_y]  # [N, T]
        per_example_loss = self.loss_fn_train(logits.transpose(1, 2), artist_ids_y)  # [N, T]
        per_example_loss *= padding_mask  # [N, T]
        loss_artist = per_example_loss.sum() / (padding_mask.sum() + 1e-6)  # []

        # combined
        loss = loss_track * self.w_track_loss + loss_artist * self.w_artist_loss

        # log
        if (self.args.local_rank == 0) & (self.state.global_step % self.args.logging_steps == 0):
            logs = {
                "loss": round(loss.item(), 4),
                "loss_track": round(loss_track.item(), 4),
                "loss_artist": round(loss_artist.item(), 4),
                "step": self.state.global_step,
            }
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)
            self.logger.info(logs)
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        loader = self.get_eval_dataloader(eval_dataset=ds)
        self.model.eval()

        num_timestamps = 0
        num_examples = 0
        loss_track = 0.0
        loss_artist = 0.0
        accuracy_track = 0.0
        accuracy_artist = 0.0
        mrr_track = 0.0
        mrr_artist = 0.0

        track_ids = torch.arange(self.model.num_tracks + self.model.num_special_tokens, device=self.args.device)
        artist_ids = self.track_id_to_artist_id[track_ids]
        keys_track = self.model.get_items_embeddings(track_ids=track_ids, artist_ids=artist_ids).T[None]

        # WARNING: будет ошибка, если tracks и artists в одной матрице эмбеддингов
        artist_ids = torch.arange(self.model.num_artists + self.model.num_special_tokens, device=self.args.device)
        keys_artist = self.model.get_items_embeddings(track_ids=None, artist_ids=artist_ids).T[None]

        pbar = tqdm.tqdm(
            desc=f"rank {self.args.local_rank}",
            total=len(ds),
            position=self.args.local_rank,
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                track_ids_x = batch["inputs"].to(self.args.device)  # [N, T]
                track_ids_y = batch["targets"].to(self.args.device)  # [N, T]
                artist_ids_x = self.track_id_to_artist_id[track_ids_x]  # [N, T]
                artist_ids_y = self.track_id_to_artist_id[track_ids_y]  # [N, T]
                batch_size = artist_ids_x.shape[0]
                num_examples += batch_size
                outputs = self.model(track_ids=track_ids_x, artist_ids=artist_ids_x)  # [N, T, D]

                padding_mask = get_padding_mask(artist_ids_x, padding_idx=self.model.padding_idx)  # [N, T]
                padding_mask = padding_mask.float().to(self.args.device)
                num_timestamps += padding_mask.sum().item()

                sequence_lengths = padding_mask.long().sum(1)
                loss_i, acc_i, mrr_i = self._get_loss_and_accuracy(outputs, keys_track, track_ids_y, padding_mask, sequence_lengths)
                loss_track += loss_i
                accuracy_track += acc_i
                mrr_track += mrr_i
                loss_i, acc_i, mrr_i = self._get_loss_and_accuracy(outputs, keys_artist, artist_ids_y, padding_mask, sequence_lengths)
                loss_artist += loss_i
                accuracy_artist += acc_i
                mrr_artist += mrr_i

                pbar.update(batch_size)
        pbar.close()

        torch.cuda.empty_cache()

        metrics = {
            "loss": (loss_track * self.w_track_loss + loss_artist * self.w_artist_loss) / num_timestamps,
            "loss_track": loss_track / num_timestamps,
            "loss_artist": loss_artist / num_timestamps,
            "accuracy_track": accuracy_track / num_timestamps,
            "accuracy_artist": accuracy_artist / num_timestamps,
            "mrr_track": mrr_track / num_examples
        }

        metrics = self._postprocess_metrics(metrics)
        return metrics

    def _get_loss_and_accuracy(self, outputs, keys, targets, mask, sequence_lengths) -> Tuple[float, float, float]:
        logits = torch.matmul(outputs, keys)  # [N, T, V]
        per_example_loss = self.loss_fn_valid(logits.transpose(1, 2), targets)  # [N, T]
        loss = (per_example_loss * mask).sum().item()
        preds = logits.argmax(-1)  # [N, T]
        accuracy = (torch.eq(targets, preds).float() * mask).sum().item()

        batch_size = logits.shape[0]
        xs = torch.arange(batch_size).to(self.args.device)
        last_item_idx = sequence_lengths - 1  # [N]
        preds_last = (logits[xs, last_item_idx, :] * -1.0).argsort(-1)  # [N, V]
        mrr = 0
        for i in range(logits.shape[0]):
            label = targets[i, last_item_idx[i]]
            for j in range(self.k):
                if preds_last[i, j] == label:
                    mrr += 1.0 / (j + 1.0)
                    break
        return loss, accuracy, mrr


class TrainerTracksOnlyFullVocab(BaseTrainer):
    """
    Языковая модель по трекам. Лосс считается по всем трекам для каждой пары (юзер, трек).
    Плюсы:
    * простота реализации
    * гипотеза: в случае с треками кажется, что истинное распределение вероятности по всем трекам
      не совсем вырожденное: юзер может равновероятно лайкнуть более одного трека.
      А значит, если учить классическим образом (просить модель максимизировать скор только на одном треке),
      то есть риск получить очень шумный сигнал -> долгая сходимость.
      Частично эту проблему можно порешать с label smoothing.
    Минусы:
    * очень вычислительно дорого: для получения логитов следующего трека для одной пары (юзер, трек)
      нужно сделать DV произведений: [1, D] x [D, V], где V порядка сотен тысяч. Оказалось,
      что учить такую штуку нужно несколько дней.

    В подходе с сэмплированием негативов (TrainerV2) требовалось сделать DM произведений:
    [1, D] x [D, M], где M порядка сотен.
    Плюсы:
    * быстрее в V / M раз
    * вероятность иметь FN снижается в V / M
    Минусы:
    * дорого по памяти: для хранения эмбеддингов-кандидатов нужно аллоцировать место под
      N x T x M x D float{16/32}. Достаточно сравнительно небольших значений M и D,
      чтоб это стало также дорого по памяти, как текущий подход, где финальные логиты занимают
      N x T x V.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        track_ids_x = inputs["inputs"]  # [N, T]
        track_ids_y = inputs["targets"]  # [N, T]

        queries = model(track_ids=track_ids_x, artist_ids=None)  # [N, T, D]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        keys = model.track_emb.weight  # [V, D]

        # [N, T, D] * [1, D, V] = [N, T, V]
        logits = torch.matmul(queries, keys.T[None])  # [N, T, V]

        # compute per-example loss
        # * class dimention must be the second (in contrast to SparseCategoricalCrossEntropy in tf)
        # * output has the same shape as labels
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        per_example_loss = self.loss_fn_train(logits.transpose(1, 2), track_ids_y)  # [N, T]

        # mask loss
        padding_mask = get_padding_mask(track_ids_x, padding_idx=model.padding_idx)
        padding_mask = padding_mask.float()
        per_example_loss *= padding_mask  # [N, T]
        loss = per_example_loss.sum() / (padding_mask.sum() + 1e-6)  # []
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        loader = self.get_eval_dataloader(eval_dataset=ds)
        self.model.eval()

        num_pairs = 0
        num_examples = 0
        total_loss = 0.0
        accuracy = 0.0
        accuracy_last = 0.0
        mrr_last = 0.0
        pbar = tqdm.tqdm(
            desc=f"rank {self.args.local_rank}",
            total=len(ds),
            position=self.args.local_rank,
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                track_ids_x = batch["inputs"].to(self.args.device)  # [N, T]
                track_ids_y = batch["targets"].to(self.args.device)  # [N, T]
                batch_size = track_ids_x.shape[0]
                num_examples += batch_size
                outputs = self.model(track_ids=track_ids_x, artist_ids=None)  # [N, T, D]
                logits = torch.matmul(outputs, self.model.track_emb.weight.T[None])  # [N, T, V]
                per_example_loss = self.loss_fn_valid(logits.transpose(1, 2), track_ids_y)  # [N, T]
                padding_mask = get_padding_mask(track_ids_x, padding_idx=self.model.padding_idx)  # [N, T]
                padding_mask = padding_mask.float()
                padding_mask = padding_mask.to(self.args.device)
                per_example_loss *= padding_mask
                total_loss += per_example_loss.sum().item()
                num_pairs += padding_mask.sum().item()

                indices = logits.argmax(-1)  # [N, T]
                accuracy += (torch.eq(track_ids_y, indices).float() * padding_mask).sum().item()

                xs = torch.arange(batch_size).to(self.args.device)
                sequence_lengths = padding_mask.sum(1).long()
                last_item_idx = sequence_lengths - 1  # [N]
                preds_last = (logits[xs, last_item_idx, :] * -1.0).argsort(-1)  # [N, V]
                for i in range(batch_size):
                    label = track_ids_y[i, last_item_idx[i]]
                    for j in range(self.k):
                        if preds_last[i, j] == label:
                            mrr_last += 1.0 / (j + 1.0)
                            if j == 0:
                                accuracy_last += 1
                            break
                pbar.update(batch_size)
        pbar.close()

        torch.cuda.empty_cache()

        metrics = {
            "loss": total_loss / num_pairs,
            "accuracy": accuracy / num_pairs,
            "accuracy_last": accuracy_last / num_examples,
            "mrr_last": mrr_last / num_examples
        }

        metrics = self._postprocess_metrics(metrics)
        return metrics
