import numpy as np
import os
import time

from multiprocessing import Process, Queue, Value


def _worker_fn(dataset, batches_remaining, index_queue, 
               sampler, batch_size, seed=None):
    while batches_remaining.value > 0:
        indices = sampler(batch_size)
        index_queue.put(indices)
        batches_remaining.value = batches_remaining.value - 1


class IndexIterator(object):
    def __init__(self, dataset, batch_size, sampler, n_workers=0, epochs=1,
                 return_last=False, seed=None, sample_test=False):
        self.n_workers = n_workers
        self.return_last = return_last
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.items_returned = 0
        self.epochs = epochs
        self.dataset_length = len(self.dataset)
        self.sampler = sampler
        if n_workers > 0:
            self.batches_remaining = Value('i', len(self))
            self.index_queue = Queue()
            self.workers = [Process(target=_worker_fn, args=(self.dataset,
                                                             self.batches_remaining,
                                                             self.index_queue,
                                                             self.sampler,
                                                             self.batch_size,
                                                             None if self.seed is None else self.seed + i))
                            for i in range(n_workers)]
            for p in self.workers:
                p.start()
        
    def __len__(self):
        n = len(self.dataset) // self.batch_size * self.epochs
        if len(self.dataset) % self.batch_size == 0:
            return n
        else:
            if self.return_last:
                return n + 1
            else:
                return n
        
    def __iter__(self):
        return self
    
    def _get_batch(self):
        if self.n_workers > 0:
            return self.index_queue.get()
        else:
            return self.sampler(self.batch_size)
        
    def __next__(self):
        items_left = self.dataset_length * self.epochs - self.items_returned
        if items_left > self.batch_size:
            sampled = self._get_batch()
            if isinstance(sampled, np.ndarray):
                n_sampled = sampled.shape[0]
            elif isinstance(sampled, dict):
                n_sampled = sampled["target"].shape[0]
            else:
                n_sampled = self.batch_size
            self.items_returned += n_sampled
            return sampled
        elif self.return_last and items_left > 0:
            sampled = self._get_batch()
            self.items_returned += items_left
            if isinstance(sampled, dict):
                return {k:v[0:items_left] for k,v in sampled.iteritems()}
            else: 
                return sampled[0:items_left]
        else:
            self._shutdown_workers()
            raise StopIteration()
    
    next = __next__
        
    def _shutdown_workers(self):
        if self.n_workers > 0:
            for p in self.workers:
                if p is not None:
                    p.terminate()
    
    def __del__(self):
        if self.n_workers > 0:
            self._shutdown_workers()



if __name__ == '__main__':
    import recsys
    from samplers import conditional
    data = recsys.ml1m(0.)
    ave = 0
    n = 5
    sleep_time = 0.2
    for epoch in range(1):
        iterator = IndexIterator(data, 10000, conditional, n_workers=1, return_last=True, epochs=n)
        t = time.time()
        for i, idx in enumerate(iterator):
            print i
            time.sleep(sleep_time)
            pass
        extra = time.time() - t
        ave += extra - (i+1) * sleep_time
    #print idx
    #print {k:v.shape for k,v in idx.iteritems()}
    print "average time per epoch: %1.3f" % (ave / n)
