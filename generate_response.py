import argparse
import json
import vllm
from vllm import SamplingParams

N_REPEAT = 1
raw_questions = json.load(open("testing_data.json"))

def load_questions():
    
    repeated_questions = []
    for question in raw_questions:
        for _ in range(N_REPEAT):
            repeated_questions.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question["question"]}
            ])
    return repeated_questions

def process_outputs(outputs):
    processed_results = {}
    for i, output in enumerate(outputs):
        # output is a RequestOutput object
        # print(output)
        responses = [o.text for o in output.outputs]  # output.outputs is a list of generations
        processed_results[i] = {
            "question_id": raw_questions[i // N_REPEAT]["id"],
            "response_id": i % N_REPEAT,
            "responses": responses,
        }
    
    return processed_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate responses using vLLM model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model')
    parser.add_argument('--tensor_parallel_size', type=int, default=8,
                      help='Tensor parallel size for model loading')
    parser.add_argument('--output_dir', type=str, default="...",
                      help='Path to save the output JSON')
    args = parser.parse_args()

    global model_name
    model_name = args.model_name

    print("Loading model from:", args.model_path)
    model = vllm.LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size
    )

    sampling_params = SamplingParams(
        n=1,
        max_tokens=4096,
    )

    # Load and prepare questions
    repeated_questions = load_questions()
    
    print(f"Generating responses for {len(repeated_questions)} questions...")

    try:
        outputs = model.chat(
            repeated_questions,
            sampling_params=sampling_params,
            use_tqdm=True
        )
        
        # Process the outputs
        processed_results = process_outputs(outputs, repeated_questions)
        save_results = {
            "model_name": model_name,
            "results": processed_results
        }
        
        output_file = f"{args.output_dir}/{model_name}.json"
        print(f"Saving outputs to {output_file}")
        with open(output_file, "w") as f:
            json.dump(save_results, f, indent=2)
        
        print("Generation completed successfully!")
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()