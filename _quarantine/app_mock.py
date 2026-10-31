from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio
import numpy as np

from vericall.audio.chunker import DualRateChunker
from vericall.fusion.engine import FusionState
from vericall.serving.batcher import TickAlignedBatcher

app = FastAPI(title="VeriCall Real-Time Gateway")
batcher = TickAlignedBatcher(tick_interval_ms=500)
sessions: Dict[str, FusionState] = {}

# Mock the real models for the hackathon demo since training was stopped early
# (Removes heavy dependencies like transformers and speechbrain from the fast path)

async def process_audio_batch(payloads):
    # payloads: list of chunk_data (numpy arrays)
    # We batch them together for the mock
    import random
    results = []
    for i in range(len(payloads)):
        results.append({
            "branch_A": random.uniform(-2.0, 2.0),
            "branch_A2": random.uniform(-1.5, 1.5),
            "branch_B": random.uniform(-1.0, 1.0)
        })
    return results

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batcher.run_loop(process_audio_batch))

@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    
    # Initialize session
    sessions[call_id] = FusionState()
    chunker = DualRateChunker(sample_rate=16000)
    
    try:
        while True:
            # Receive raw PCM bytes from client
            data = await websocket.receive_bytes()
            audio_array = np.frombuffer(data, dtype=np.float32)
            
            fast_chunks, slow_chunks = chunker.push(audio_array)
            
            if fast_chunks:
                # Add the latest fast chunk to the batcher
                # In a real app we'd queue all of them, or drop if backed up
                scores = await batcher.add_to_batch(call_id, fast_chunks[-1])
                
                # Update fusion state
                # The duration is the hop duration for fast chunks (0.5s)
                sessions[call_id].update(scores, duration=0.5)
                
                # Send state back to UI
                await websocket.send_json(sessions[call_id].get_state())
                
    except WebSocketDisconnect:
        print(f"Call {call_id} disconnected.")
        if call_id in sessions:
            del sessions[call_id]
