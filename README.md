# Converting a LoRA Model to GGUF Format for Ollama

## Overview
This guide walks through the process of converting a LoRA fine-tuned model into GGUF format for use with Ollama. The process involves:
1. Loading and saving the model and tokenizer
2. Uploading to Hugging Face (HF)
3. Downloading the base model and LoRA adapter
4. Combining them into a single GGUF file
5. Compiling `llama.cpp` for inference
6. Creating an Ollama-compatible model file

## Step 1: Load and Save LoRA Model & Tokenizer
First, load your fine-tuned LoRA model and tokenizer, then save them:

```python
from fastllm import FastLanguageModel

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="<model_name>",
        max_seq_length=10000,
        top_p=0.3,
        top_k=100,
        dtype=None,
        load_in_4bit=True,
    ) 
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()
model.save_pretrained("./<dir/location>")
tokenizer.save_pretrained("./<dir/location>")
```

## Step 2: Upload to Hugging Face
1. Visit [Hugging Face](https://huggingface.co/)
2. Create a new model repository
3. Upload the `<dir/location>` directory

## Step 3: Convert to GGUF
Obtain GGUF versions for both base model and LoRA adapter from HF Spaces:

- Base Model: `https://huggingface.co/spaces/ggml-org/gguf-my-repo`
- LoRA Adapter: `https://huggingface.co/spaces/ggml-org/gguf-my-lora`
## 3. Converting and Merging to GGUF Using llama.cpp

The next step is to obtain GGUF versions for both the base model and your LoRA adapter. In our workflow these files are hosted on HF Spaces. For example, you might have:

- **Base Model GGUF:**  
  [https://huggingface.co/spaces/ggml-org/gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo)

- **LoRA Adapter GGUF:**  
  [https://huggingface.co/spaces/ggml-org/gguf-my-lora](https://huggingface.co/spaces/ggml-org/gguf-my-lora)


## Step 4: Download Base Model & LoRA Adapter from HF
To obtain the required files:

```sh
curl -L -o <model_name_as_on_above_links_from_step3> -H "Authorization: Bearer <HFtoken>" \
https://huggingface.co/<userName>/<model_name>+<-GGUF>/resolve/main/<model_name>.gguf

curl -L -o <model_name_as_on_above_links_from_step3_for_LORA> -H "Authorization: Bearer <HFtoken>" \
https://huggingface.co/<userName>/<model_name_for_LORA>+<-GGUF>/resolve/main/<model_name_for_LORA>.gguf
```
I suggest use the links from step 3 to figure out the neceesary curl URLS, **check below eg**

```bash
curl -L -o llama-3.1-8b-instruct-q4_k_m.gguf -H "Authorization: Bearer <token>" \
"https://huggingface.co/userName/Llama-3.1-8B-Instruct-Q4_K_M-GGUF/resolve/main/llama-3.1-8b-instruct-q4_k_m.gguf"

curl -L -o v2_3lora_model_3_1_Llama8b_Inst-f16.gguf -H "Authorization: Bearer <token>" \
"https://huggingface.co/userName/v2_3lora_model_3_1_Llama8b_Inst-F16-GGUF/resolve/main/v2_3lora_model_3_1_Llama8b_Inst-f16.gguf"
```

These commands will download:
- A quantized (q4) base GGUF model file
- A 16‑bit (f16) LoRA adapter GGUF file

## Step 5: Compile `llama.cpp`
Clone and build `llama.cpp` for running inference:

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Non-GPU Compilation
```sh
cmake -B build
cmake --build build --config Release
```

### GPU Compilation
```sh
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

## Step 6: Combine Base Model & LoRA Adapter into a Single GGUF
Execute the following command to merge the models:

```sh
./build/bin/llama-export-lora \
    -m /<full_model_location> \
    -o /<combined_model_location> \
    --lora /<lora_adapters_location> \
    --verbose
```
**Below is an eg**

```sh
./build/bin/llama-export-lora \
    -m ../llama-3.1-8b-instruct-q4_k_m.gguf \
    -o ../combined.gguf \
    --lora ../v2_3lora_model_3_1_Llama8b_Inst-f16.gguf \
    --verbose
```
Here:
- `-m` specifies the base model GGUF file.
- `--lora` specifies the LoRA adapter GGUF file.
- `-o` sets the output combined GGUF file.

**Note:** If you encounter a "No space left" error, free up disk space or quantize the model to `q8` instead of `f16`.

## Step 7: Create Ollama Model File
1. Create a `Modelfile` in the same directory as `combined.gguf`.
2. Add the following content:

```
FROM combined.gguf
```

3. Run the following command to import into Ollama:

```sh
ollama create <model_name> -f Modelfile
```
Replace `<model_name>` with your desired model name. This command instructs Ollama to create a new model based on the GGUF file specified in the Modelfile.

---
## Troubleshooting and Dynamic Considerations

- **Disk Space:**  
  If you encounter “no space” errors during any of the conversion or merging steps, consider using a machine or environment with more disk space or quantize the model to a slightly lower precision (e.g., q8 instead of f16) to reduce file sizes.

- **Quantization Methods:**  
  The guide above uses q4 (for the base model) and f16 (for the LoRA adapter) formats. Depending on your hardware and performance needs, you may choose different quantization methods. Check the supported quantization methods in the [llama.cpp quantization documentation](https://github.com/ggerganov/llama.cpp) for details.

- **Dynamic Paths and File Names:**  
  Adjust paths and file names in the commands as per your local environment. The instructions here assume that the GGUF files and the llama.cpp repository are located in directories relative to each other. Update the relative paths if needed.

- **Model Compatibility:**  
  Ensure that your base model and LoRA adapter are compatible with each other and follow the naming conventions expected by the conversion scripts. In case of any errors (such as missing keys), verify that the tokenizer and configuration files are in place.

---

## Conclusion

This documentation has outlined a full workflow to convert a LoRA‑tuned model into a GGUF file that works with Ollama:

1. **Load and save your model & tokenizer** using Unsloth’s `FastLanguageModel` and the `save_pretrained` methods.
2. **Upload the model directory** to Hugging Face.
3. **Download pre‑converted GGUF files** (for both the base model and LoRA adapter) using curl.
4. **Clone and build llama.cpp**, then merge the GGUF files with `llama-export-lora`.
5. **Create a Modelfile** referencing the combined GGUF file and import it into Ollama.

---

By following these steps, you can transform your LoRA weights into a fully combined GGUF model that is ready for inference in Ollama. This guide is adaptable to many dynamic environments—simply adjust file paths, quantization parameters, and system configurations as needed.


Feel free to reach out on [LinkedIn](https://www.linkedin.com/in/hrishk/) or raise an issue or comment.
