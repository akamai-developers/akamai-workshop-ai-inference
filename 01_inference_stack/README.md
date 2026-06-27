# Module 1: The Inference Stack and Runtime Landscape

**Goal:** reach the GPU you own, see an agent run on it, and learn the map of the stack
before you start measuring it.

**Built from:** old `00_connect_and_verify` plus `01_renting_vs_owning`, with the runtime
landscape and the request trace added.

## Sections
1. Setup
2. Reach the server you own (settings, pods, GPU, first request)
3. The agent on your endpoint
4. Renting versus owning, with your own numbers
5. What a price tag hides
6. The runtime landscape
7. Trace one request, end to end

## Needs
- A vLLM endpoint in `VLLM_HOST`, and `kubectl` scoped to your namespace

## Files
- `01_inference_stack.ipynb` the lab
- `images/01_inference_stack_architecture.png` the diagram (to render)

_Status: notebook drafted. Diagram and live run pending._
