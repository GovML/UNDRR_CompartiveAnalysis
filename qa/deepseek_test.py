import logging
import sys
from deepseek_client import DeepSeekClient

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_basic_generation():
    """Test basic text generation with DeepSeek."""
    print("\n=== Testing Basic Generation ===")
    
    client = DeepSeekClient(model_name="deepseek-r1:7b", temperature=0.6)
    
    prompt = "Explain Newton's second law of motion in simple terms."
    
    print(f"Prompt: {prompt}")
    print("\nGenerating response...")
    
    response = client.generate(prompt)
    
    print("\nResponse:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    
    return response is not None and len(response) > 0

def test_json_generation():
    """Test generating and extracting JSON."""
    print("\n=== Testing JSON Generation ===")
    
    client = DeepSeekClient(model_name="deepseek-r1:7b", temperature=0.6)
    
    system_prompt = "You are a helpful assistant that always responds with valid JSON."
    prompt = """
    Provide information about three planets in our solar system.
    Format your response as a JSON array with objects having these properties:
    - name: The name of the planet
    - position: Position from the sun
    - diameter: Diameter in kilometers
    - fun_fact: An interesting fact about the planet
    """
    
    print(f"Prompt: {prompt}")
    print("\nGenerating response...")
    
    # First get the raw response
    raw_response = client.generate(prompt, system_prompt, raw_response=True)
    
    print("\nRaw response:")
    print("-" * 40)
    print(raw_response[:500] + "..." if len(raw_response) > 500 else raw_response)
    print("-" * 40)
    
    # Now try to extract JSON
    try:
        json_data = client.extract_json(raw_response)
        print("\nExtracted JSON:")
        print("-" * 40)
        import json
        print(json.dumps(json_data, indent=2))
        print("-" * 40)
        return True
    except ValueError as e:
        print(f"JSON extraction failed: {e}")
        return False

def test_streaming():
    """Test streaming response generation."""
    print("\n=== Testing Streaming ===")
    
    client = DeepSeekClient(model_name="deepseek-r1:7b")
    
    prompt = "Write a short poem about artificial intelligence."
    
    print(f"Prompt: {prompt}")
    print("\nStreaming response:")
    print("-" * 40)
    
    # Collect the chunks to verify we got something
    chunks = []
    
    try:
        for i, chunk in enumerate(client.generate(prompt, stream=True)):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
            # Break after some chunks to keep test short
            if i > 20:
                print("\n\n[Truncated for brevity]")
                break
        print("\n" + "-" * 40)
        return len(chunks) > 0
    except Exception as e:
        print(f"\nStreaming failed: {e}")
        return False

if __name__ == "__main__":
    print("DEEPSEEK CLIENT TEST")
    print("=" * 40)
    
    # Track test results
    results = {}
    
    try:
        results["basic_generation"] = test_basic_generation()
        results["json_generation"] = test_json_generation()
        results["streaming"] = test_streaming()
        
        # Print summary
        print("\nTEST SUMMARY")
        print("=" * 40)
        for test, passed in results.items():
            print(f"{test}: {'PASSED' if passed else 'FAILED'}")
        
        # Overall result
        if all(results.values()):
            print("\nAll tests PASSED! The DeepSeek client is working correctly.")
        else:
            print("\nSome tests FAILED. Check the logs for details.")
    
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        import traceback
        traceback.print_exc()