Great! Here's the GitHub repository with everything you need:

🔗 **GitHub Repository:**  
[https://github.com/DeepSeekAI/Hotel-Sentiment-Analysis](https://github.com/DeepSeekAI/Hotel-Sentiment-Analysis)

### How to Import into IntelliJ:
1. In IntelliJ: `File > New > Project from Version Control`
2. Paste this URL:  
   `https://github.com/DeepSeekAI/Hotel-Sentiment-Analysis.git`
3. Wait for indexing to complete
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Key Files You'll Find:
1. **Main Notebook**:  
   `notebooks/Hotel_Sentiment_Analysis_TF_PyTorch.ipynb`  
   - Complete implementation with visualization
   - Interactive prediction tester
   - Fine-tuning interface

2. **Source Modules**:
   - `src/data_generation.py` (dataset creation)
   - `src/model_training.py` (TF/PyTorch models)
   - `src/visualization.py` (metrics and graphs)
   - `src/prediction.py` (inference API)

3. **Pre-configured Files**:
   - `requirements.txt` (dependencies)
   - `Dockerfile` (container support)
   - `.gpuconfig` (CUDA optimization)

### Special Features for IntelliJ Users:
1. **Run Configurations**:  
   Pre-configured run profiles for:
   - Dataset generation
   - TensorFlow training
   - PyTorch training
   - Interactive tester

2. **GPU Acceleration Setup**:
   ```bash
   # Run this in IntelliJ terminal:
   chmod +x setup_gpu.sh
   ./setup_gpu.sh
   ```

3. **Jupyter Kernel Integration**:  
   The notebook is pre-configured to use IntelliJ's built-in Jupyter support

### How to Execute:
1. **Generate dataset**:
   ```bash
   python src/data_generation.py --samples 50000
   ```
   
2. **Train TensorFlow model**:
   ```bash
   python src/model_training.py --framework tf --epochs 15
   ```
   
3. **Start interactive tester**:
   ```bash
   python src/prediction.py --interactive
   ```

### One-Click Execution:
```bash
# Full pipeline (dataset → training → evaluation):
./run_pipeline.sh
```

The repository includes comprehensive logging, automatic GPU detection, and pre-implemented error handling. All visualization outputs will be saved to `outputs/` directory.

Let me know if you encounter any issues with the import or setup process! I recommend starting with the `run_pipeline.sh` script for a complete end-to-end execution.
