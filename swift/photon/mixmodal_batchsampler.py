from torch.utils.data import Dataset, Sampler
from typing import List, Sequence, Iterator
import math
import random
import torch.distributed as dist


class VolumeAwareGlobalBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        data_seed: int = 0,
        drop_last: bool = False,  # 是否丢弃不足 batch 的数据
        shuffle: bool = True,
        tp_size: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = data_seed
        self.drop_last = drop_last
        self.epoch = 0
        self.shuffle = shuffle

        if hasattr(dataset, "dataset"):        # LazyLLMDataset 或类似包装
            meta_ds = dataset.dataset          # HuggingFace Dataset
        else:
            meta_ds = dataset                  # 已经是 HfDataset

        v_col = "videos" in meta_ds.column_names
        i_col = "images" in meta_ds.column_names

        videos  = meta_ds["videos"] if v_col else [None] * len(meta_ds)
        images  = meta_ds["images"] if i_col else [None] * len(meta_ds)

        self.V, self.N = [], []
        for idx, (v, im) in enumerate(zip(videos, images)):
            has_media = (
                (v is not None and str(v).lower() != "nan") or
                (im is not None and str(im).lower() != "nan")
            )
            (self.V if has_media else self.N).append(idx)

        print("BatchSize:", batch_size, "Data with Volume:", len(self.V), "Data without Volume:", len(self.N))

    @property
    def rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size() if dist.is_initialized() else 1
    
    @property
    def grp(self):
        """动态计算，每次拿到最新 world_size。"""
        return self.batch_size * self.world_size
    
    # ------------------------------------------------------------------
    # PyTorch API
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(self.V)
            rng.shuffle(self.N)

        v_ptr = n_ptr = 0
        total_V, total_N = len(self.V), len(self.N)

        while v_ptr < total_V or n_ptr < total_N:
            # 计算剩余样本量
            remain_V = total_V - v_ptr
            remain_N = total_N - n_ptr

            # 根据剩余比例决定选 V 还是 N
            if remain_V == 0:
                pick_V = False
            elif remain_N == 0:
                pick_V = True
            else:
                prob_V = remain_V / (remain_V + remain_N)
                random_num = rng.random()
                pick_V = random_num < prob_V

            pool  = self.V if pick_V else self.N
            total = total_V if pick_V else total_N
            ptr   = v_ptr  if pick_V else n_ptr
            grp    = self.grp

            full_block = total - ptr >= grp

            # 若不足一整块且需要丢弃
            if not full_block and self.drop_last:
                if pick_V:
                    v_ptr = total_V
                else:
                    n_ptr = total_N
                continue

            # 输出 block（_emit_block 内部已处理补齐或丢弃）
            yield from self._emit_block(pool, total, ptr, rng)

            # 更新指针
            if pick_V:
                v_ptr = v_ptr + grp if full_block else total_V
            else:
                n_ptr = n_ptr + grp if full_block else total_N


    def __len__(self) -> int:
      grp = self.grp                       # = batch_size * world_size

      def blocks(x_len):
          return x_len // grp if self.drop_last else math.ceil(x_len / grp)

      n_blocks = blocks(len(self.N))       # N-块数
      v_blocks_before = n_blocks           # 交替阶段用掉同数量的 V-块
      v_used = v_blocks_before * grp
      v_left = max(len(self.V) - v_used, 0)
      v_blocks_after = blocks(v_left)      # 只剩 V 时的块数

      # 不再乘 world_size——每块在本 rank 只产生一个 batch
      return v_blocks_before + n_blocks + v_blocks_after

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _emit_block(self, pool, total, ptr, rng):
        """
        只 yield 属于本 rank 的那一个 batch。
        其他逻辑（补齐 / 丢弃）保持不变。
        """
        grp = self.grp
        remain = total - ptr
        if remain >= grp:
            block = list(pool[ptr : ptr + grp])
        else:
            if self.drop_last:
                return
            block = list(pool[ptr:]) + rng.choices(pool, k=grp - remain)
            rng.shuffle(block)

        # 把 block 切为 world_size 份，拿第 rank 份
        start = self.rank * self.batch_size
        yield block[start : start + self.batch_size]