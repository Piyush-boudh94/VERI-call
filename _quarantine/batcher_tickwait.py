import asyncio
from typing import List, Dict, Any
import time

class TickAlignedBatcher:
    """
    Tick-aligned cross-call batcher to ensure high GPU utilization.
    Snaps all calls to a shared 500ms grid to form maximal batch sizes.
    """
    def __init__(self, tick_interval_ms: int = 500):
        self.tick_interval_ms = tick_interval_ms
        self.tick_interval_sec = tick_interval_ms / 1000.0
        self.queue = []
        self._lock = asyncio.Lock()
        
    async def add_to_batch(self, call_id: str, payload: Any):
        """
        Adds a payload (e.g. an audio chunk) to the current batch.
        Returns a Future that will resolve with the model's output for this payload.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        async with self._lock:
            self.queue.append((call_id, payload, future))
            
        return await future
        
    async def run_loop(self, process_batch_fn):
        """
        Background task that wakes up every tick, gathers everything in the queue, 
        and processes it in one batch.
        """
        while True:
            # Align to next tick
            now = time.time()
            next_tick = (int(now / self.tick_interval_sec) + 1) * self.tick_interval_sec
            sleep_time = next_tick - now
            await asyncio.sleep(sleep_time)
            
            async with self._lock:
                if not self.queue:
                    continue
                
                # Pop all items from the queue
                batch = self.queue
                self.queue = []
                
            # Unzip the batch
            call_ids = [item[0] for item in batch]
            payloads = [item[1] for item in batch]
            futures = [item[2] for item in batch]
            
            try:
                # Process the batch using the provided function
                # process_batch_fn should take a list of payloads and return a list of results
                results = await process_batch_fn(payloads)
                
                # Resolve the futures
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
