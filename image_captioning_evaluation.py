import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
import evaluate
from tqdm import tqdm

def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print("Loading BLIP image captioning model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    # Load dataset
    print("Loading MSCOCO test dataset...")
    dataset = load_dataset("nlphuji/flickr30k")#nlphuji/mscoco_2014_5k_test_image_text_retrieval")
    test_dataset = dataset["test"]
    print(f"Loaded {len(test_dataset)} test samples")

    # Load evaluation metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Generate captions and collect references
    predictions = []
    references = []

    print("Generating captions...")
    for idx, item in enumerate(tqdm(test_dataset)):
        # Get image and reference captions
        image = item["image"]
        image_captions = item["caption"]
        
        # Process image and generate caption with the model
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=50)
        
        # Decode the generated caption
        generated_caption = processor.decode(output[0], skip_special_tokens=True)
        
        predictions.append(generated_caption)
        references.append(image_captions)
        
        # Print a few examples
        if idx < 3:
            print(f"\nExample {idx}:")
            print(f"Generated: {generated_caption}")
            print(f"References: {image_captions}")

    # Evaluate using metrics
    print("\nCalculating evaluation metrics...")
    
    # BLEU
    bleu_score = bleu.compute(predictions=predictions, references=references)
    
    # ROUGE - need single reference per prediction for rouge
    rouge_score = rouge.compute(
        predictions=predictions,
        references=[refs[0] for refs in references]  # Using first reference caption
    )
    
    # METEOR
    meteor_score = meteor.compute(predictions=predictions, references=references)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"BLEU score: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-1 score: {rouge_score['rouge1']:.4f}")
    print(f"ROUGE-2 score: {rouge_score['rouge2']:.4f}")
    print(f"METEOR score: {meteor_score['meteor']:.4f}")

if __name__ == "__main__":
    main()