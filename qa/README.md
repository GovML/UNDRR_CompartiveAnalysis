# DeepSeek Client for Ollama

A Python client for interacting with local DeepSeek models through Ollama.

## Installation

1. **Install Ollama**

   Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)

2. **Pull the DeepSeek model**

   ```bash
   # This will download the model and start a chat session with it
   ollama run deepseek-r1:7b
   ```

   In the code we use the `DeepSeek-R1-Distill-Qwen-7B` model. You can also pull other DeepSeek models available in Ollama.

3. **Install Python dependencies**

   ```bash
   pip install ollama tenacity
   ```

## Running the Test Script

1. Run the test script:

   ```bash
   python qa/deepseek_test.py
   ```

   If the model is set up correctly, you should see:

   ```
   TEST SUMMARY
   ========================================
   basic_generation: PASSED
   json_generation: PASSED
   streaming: PASSED

   All tests PASSED! The DeepSeek client is working correctly.
   ```

## Troubleshooting

If you encounter issues:

1. **Ollama not running**:

   - Make sure Ollama is running with `ollama serve`
   - Check if you can run basic commands like `ollama list`

2. **Model not found**:

   - Verify you've pulled the correct model with `ollama list`
   - The model name should be exactly "deepseek-r1:7b" as used in the code
   - If needed, pull the model again with `ollama run deepseek-r1:7b`

3. **Python errors**:
   - Ensure you've installed all dependencies
   - Check that your Python version is 3.7 or higher
