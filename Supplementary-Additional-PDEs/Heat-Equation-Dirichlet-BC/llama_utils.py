import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_processing import unscale_2d_array, deserialize_2d_integers, extract_training_and_test

def load_model_and_tokenizer(model_name):
    """
    Loads the Llama model and its corresponding tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer


def build_good_and_bad_tokens(tokenizer, extra_allowed_tokens=[","]):
    """
    Build the sets of allowed and disallowed token IDs
    Allowed tokens include:
      - All three-digit strings "000" to "999"
      - Each string in extra_allowed_tokens
    Raises:
      ValueError if any allowed string tokenizes to a sequence length != 1.
    Returns:
      good_tokens (list of int): IDs for allowed tokens
      bad_tokens (list of int): IDs for all other tokens
    """
    good_tokens = set()
    # Add all three-digit numbers
    for i in range(1000):
        token_str = f"{i:03d}"
        token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Expected exactly one token for '{token_str}', got {token_ids}")
        good_tokens.add(token_ids[0])
    # Add any extra allowed tokens
    for token_str in extra_allowed_tokens:
        token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Expected exactly one token for '{token_str}', got {token_ids}")
        good_tokens.add(token_ids[0])
    # Everything not allowed is considered bad
    vocab_size = tokenizer.vocab_size
    bad_tokens = [i for i in range(vocab_size) if i not in good_tokens]
    # Add the EOS token
    if tokenizer.eos_token_id is not None:
      bad_tokens.append(tokenizer.eos_token_id)

    return list(good_tokens), bad_tokens


def generate_text_multiple(prompt, model, tokenizer, Nx):
    """
    Generates new text and returns the probability for each newly generated token.
    Input:
      prompt: The input prompt
      model: The loaded language model
      tokenizer: The corresponding tokenizer
      Nx: A number used to determine max_new_tokens (here, 2*Nx-1 tokens will be generated)
    Output:
      generated_text: The newly generated text (tokens after the prompt)
      output: Hugging Face generation output object, which includes per-token probabilities.
    """
    max_new_tokens = 2 * Nx - 1
    good_tokens, bad_tokens = build_good_and_bad_tokens(tokenizer, extra_allowed_tokens=[","])
    bad_words_ids = [[t] for t in bad_tokens]
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=bad_words_ids,
        renormalize_logits=True,
        output_scores=True,
        return_dict_in_generate=True,
    )
    # The generated_ids include both prompt and newly generated tokens
    # Separate out just the portion that is newly generated
    prompt_length = inputs.input_ids.shape[1]
    generated_ids = output.sequences[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, output


# Helper function to run a single LLM prediction sequence
def _run_llm_prediction(full_serialized_data, input_time_steps, number_of_future_predictions,
                        model, tokenizer, Nx, settings, vmin, vmax):
    all_rows_scaled = deserialize_2d_integers(full_serialized_data, settings)
    train_serial, _ = extract_training_and_test(full_serialized_data, input_time_steps, settings)
    if not train_serial.endswith(settings.time_sep):
        train_serial += settings.time_sep
    current_prompt = train_serial
    predicted_rows_unscaled = []
    max_diffs = []
    rmses = []
    for step_idx in range(number_of_future_predictions):
        gt_idx = input_time_steps + step_idx
        if gt_idx >= all_rows_scaled.shape[0]:
            # Stop if we exceed the available ground truth
            break
        next_line, _ = generate_text_multiple(current_prompt, model, tokenizer, Nx)
        next_line = next_line.strip()
        predicted_scaled_2d = deserialize_2d_integers(next_line, settings)
        predicted_unscaled_2d = unscale_2d_array(predicted_scaled_2d, vmin, vmax)
        predicted_rows_unscaled.append(predicted_unscaled_2d)
        current_prompt += next_line + settings.time_sep
        # Calculate errors
        gt_scaled = all_rows_scaled[gt_idx]
        gt_unscaled = unscale_2d_array(gt_scaled[np.newaxis, :], vmin, vmax)[0]
        pred_unscaled = predicted_unscaled_2d[0]
        max_diff = np.max(np.abs(pred_unscaled - gt_unscaled))
        rmse = np.sqrt(np.mean((pred_unscaled - gt_unscaled)**2))
        max_diffs.append(max_diff)
        rmses.append(rmse)
    
    return predicted_rows_unscaled, max_diffs, rmses


# Run LLM predictions with multiple seeds
def llm_multi_predictions(full_serialized_data, input_time_steps, number_of_future_predictions, 
                          model, tokenizer, Nx, settings, vmin, vmax, n_seeds):
    # Store predictions and errors for each seed
    predictions_by_seed = []
    all_seeds_max_diffs = []
    all_seeds_rmses = []
    for seed in tqdm(range(n_seeds)):
        if n_seeds > 1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        # Generate predictions for current seed
        preds, max_diffs, rmses = _run_llm_prediction(full_serialized_data, input_time_steps, 
                                                      number_of_future_predictions, model, tokenizer, Nx, 
                                                      settings, vmin, vmax)
        predictions_by_seed.append(preds)
        all_seeds_max_diffs.append(max_diffs)
        all_seeds_rmses.append(rmses)
    # If only one seed, return results directly
    if n_seeds == 1:
        return all_seeds_max_diffs[0], all_seeds_rmses[0], predictions_by_seed[0]
    # Otherwise, average results across seeds
    max_steps = min(len(diffs) for diffs in all_seeds_max_diffs)
    avg_max_diffs = []
    avg_rmses = []
    std_max_diffs = [] 
    std_rmses = []
    for step in range(max_steps):
        # Average errors for this step across all seeds
        step_max_diffs = [seed_diffs[step] for seed_diffs in all_seeds_max_diffs]
        step_rmses = [seed_rmses[step] for seed_rmses in all_seeds_rmses]
        avg_max_diffs.append(np.mean(step_max_diffs))
        avg_rmses.append(np.mean(step_rmses))
        # Calculate standard deviations
        std_max_diffs.append(np.std(step_max_diffs, ddof=1))
        std_rmses.append(np.std(step_rmses, ddof=1))
    averaged_predictions = []
    for step in range(min(number_of_future_predictions, max_steps)):
        step_preds = [seed_preds[step] for seed_preds in predictions_by_seed 
                     if step < len(seed_preds)]
        avg_pred = np.mean(step_preds, axis=0)
        averaged_predictions.append(avg_pred)
    
    return avg_max_diffs, avg_rmses, averaged_predictions, std_max_diffs, std_rmses


# Helper function to run LLM-based single‐step prediction with in‐loop retry for smaller models
def _run_llm_prediction_smaller_model(full_serialized_data, input_time_steps, number_of_future_predictions, 
                                      model, tokenizer, Nx, settings, vmin, vmax, max_retries: int = 10):
    all_rows_scaled = deserialize_2d_integers(full_serialized_data, settings)
    train_serial, _ = extract_training_and_test(full_serialized_data, input_time_steps, settings)
    if not train_serial.endswith(settings.time_sep):
        train_serial += settings.time_sep
    current_prompt = train_serial
    predicted_rows_unscaled = []
    max_diffs = []
    rmses = []
    for step_idx in range(number_of_future_predictions):
        gt_idx = input_time_steps + step_idx
        if gt_idx >= all_rows_scaled.shape[0]:
            break
        # Retry this single prediction until we get a (1, Nx) array
        valid_step = False
        for attempt in range(max_retries):
            next_line, _ = generate_text_multiple(current_prompt, model, tokenizer, Nx)
            next_line = next_line.strip()
            pred_scaled_2d = deserialize_2d_integers(next_line, settings)
            if pred_scaled_2d.shape == (1, Nx):
                valid_step = True
                break
            else:
                print(
                    f" Step {step_idx} attempt {attempt} failed: Got shape {pred_scaled_2d.shape}, expected second dim to be {Nx}"
                )
        if not valid_step:
            print(
                f"  Warning: step {step_idx} failed all {max_retries} retries; using last output"
            )
        # Unscale and record prediction
        predicted_unscaled_2d = unscale_2d_array(pred_scaled_2d, vmin, vmax)
        predicted_rows_unscaled.append(predicted_unscaled_2d)
        current_prompt += next_line + settings.time_sep
        # Compute error against ground truth
        gt_scaled = all_rows_scaled[gt_idx]
        gt_unscaled = unscale_2d_array(gt_scaled[np.newaxis, :], vmin, vmax)[0]
        pred_unscaled = predicted_unscaled_2d[0]
        max_diff = np.max(np.abs(pred_unscaled - gt_unscaled))
        rmse = np.sqrt(np.mean((pred_unscaled - gt_unscaled) ** 2))
        max_diffs.append(max_diff)
        rmses.append(rmse)

    return predicted_rows_unscaled, max_diffs, rmses


# Run multiple seeds of LLM predictions
def llm_multi_predictions_smaller_model(full_serialized_data, input_time_steps, number_of_future_predictions,
                                        model, tokenizer, Nx, settings, vmin, vmax, n_seeds):
    predictions_by_seed = []
    all_seeds_max_diffs = []
    all_seeds_rmses = []
    for seed in tqdm(range(n_seeds)):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Single call per seed
        preds, max_diffs, rmses = _run_llm_prediction_smaller_model(full_serialized_data, input_time_steps,
                                                                    number_of_future_predictions, model, 
                                                                    tokenizer, Nx, settings, vmin, vmax,
                                                                    max_retries=10)
        predictions_by_seed.append(preds)
        all_seeds_max_diffs.append(max_diffs)
        all_seeds_rmses.append(rmses)
    # Aggregate errors across seeds
    max_steps = min(len(d) for d in all_seeds_max_diffs)
    avg_max_diffs, avg_rmses = [], []
    std_max_diffs, std_rmses = [], []
    for step in range(max_steps):
        vals_max = [d[step] for d in all_seeds_max_diffs]
        vals_rmse = [r[step] for r in all_seeds_rmses]
        avg_max_diffs.append(np.mean(vals_max))
        avg_rmses.append(np.mean(vals_rmse))
        std_max_diffs.append(np.std(vals_max, ddof=1))
        std_rmses.append(np.std(vals_rmse, ddof=1))
    averaged_predictions = []
    for step in range(min(number_of_future_predictions, max_steps)):
        step_preds = [seed_preds[step] for seed_preds in predictions_by_seed if step < len(seed_preds)]
        averaged_predictions.append(np.mean(step_preds, axis=0))

    return avg_max_diffs, avg_rmses, averaged_predictions, std_max_diffs, std_rmses