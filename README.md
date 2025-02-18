### This project provides a FastAPI-based service for generating embeddings using one of the top open-source models from the MTEB Hugging Face leaderboard. It wraps the model in an API, optimizing it for high-performance inference on either GPU (if available) or CPU. The solution is designed for seamless integration with vector databases and includes a client wrapper for easy usage.

Key Features
•	Top-performing Model: Utilizes one of the best open-source models Mistral from the MTEB Hugging Face leaderboard for high-quality embeddings.
•	GPU and CPU Support: Automatically switches between GPU and CPU, leveraging torch.inference_mode() for optimal performance.
•	Batch Processing: Processes data in batches, significantly reducing embedding time.
•	Customizable Client Wrapper: Includes a client-side wrapper that interacts with the API to embed either single queries or multiple documents.
•	Vector Database Integration: Designed for direct integration with popular vector databases, enabling efficient storage and retrieval of embeddings.
