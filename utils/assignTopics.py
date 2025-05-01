import torch 

def get_reviews_labels_LLM_3shots(review, topics, llm_client, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful customer reviews expert that identifies the main topics in a review.\n"
                "Provide the output as a comma-separated list of topics. The first letter of the topics should always be capitalized.\n"
            ),
        },
        # Example 1
        {
            "role": "user",
            "content": (
                "Read the following review and associate the topics mentioned implicitly or explicitly in the review.\n"
                "Only answer with the topics that are mentioned in the review. Example: ['price', 'quality']. \n"
                "If you cannot identify any topics, just return '[]' \n"
                "Review: 'The product arrived quickly, and the packaging was great. However, the price is too high.'\n"
                "Topics: ['Delivery', 'Packaging', 'Price', 'Quality']. DO NOT write any topics outside of this list. \n"
                "Topics mentioned within the review:"
            ),
        },
        {
            "role": "assistant",
            "content": "['Delivery', 'Packaging', 'Price']",
        },
        # Example 2
        {
            "role": "user",
            "content": (
                "Read the following review and associate the topics mentioned implicitly or explicitly in the review.\n"
                "Only answer with the topics that are mentioned in the review. Example: ['price', 'quality']. \n"
                "If you cannot identify any topics, just return '[]' \n"
                "Review: 'I love how comfortable these shoes are! They fit perfectly, and the material feels premium. Good price.'\n"
                "Topics: ['Comfort', 'Fit', 'Material', 'Design', 'Value']. DO NOT write any topics outside of this list.\n"
                "Topics mentioned within the review:"
            ),
        },
        {
            "role": "assistant",
            "content": "['Comfort', 'Fit', 'Material', 'Value']",
        },
        # Example 3
        {
            "role": "user",
            "content": (
                "Read the following review and associate the topics mentioned implicitly or explicitly in the review.\n"
                "Only answer with the topics that are mentioned in the review. Example: ['price', 'quality']. \n"
                "If you cannot identify any topics, just return '[]' \n"
                "Review: 'The app crashes frequently and is very slow. It needs major improvements.'\n"
                "Topics: ['Performance', 'Usability', 'Design']. DO NOT write any topics outside of this list. \n"
                "Topics mentioned within the review:"
            ),
        },
        {
            "role": "assistant",
            "content": "['Performance', 'Usability']",
        },
        {
            "role": "user",
            "content": (
                "Read the following review and associate the topics mentioned implicitly or explicitly in the review.\n"
                "Only answer with the topics that are mentioned in the review. Example: ['Price', 'Quality']. \n"
                "If you cannot identify any topics, just return '[]' \n"
                f"Review: '{review}' \n"
                f"Topics: {topics}. DO NOT write any topics outside of this list. \n"
                f"Topics mentioned within the review:"
            ),
        },
    ]

    response = ' '
    model = "gpt-4o" 
    # Generate the topic word using the language model
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=30,
        temperature=0.4,
        n=1,
        stop=None,
    )

    # Extract and return the topic word
    return response.choices[0].message.content.strip()


def get_reviews_labels_deBERTa(tokenizer, model, device, review: str, topics: list[str], threshold: float = 0.5) -> list[str]:
    # Build the single‐pass input that your model was trained on
    input_text = f"Review: {review} | Topics: {', '.join(topics)}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)         # shape: (num_labels,)
        probs  = torch.sigmoid(logits).cpu()              # to CPU for numpy ops if desired

    # Only keep topics whose probability ≥ threshold
    assigned = []
    for topic in topics:
        idx = model.config.label2id.get(topic, None)
        if idx is not None and probs[idx] >= threshold:
            assigned.append(topic)
    return assigned


def get_reviews_labels_flanT5(
    tokenizer,
    model,
    device,
    review: str,
    topics: list[str],
    max_length: int = 64,
    num_beams: int = 4
) -> list[str]:
    """
    Given a single `review` and a list of candidate `topics`,
    use the fine-tuned Flan-T5 model to pick which topics apply.
    Returns a list of assigned topics.
    """
    # build a prompt that our fine-tuned T5 expects
    prompt = (
        "Classify this review into one or more of the following topics: "
        f"{', '.join(topics)}.\n\n"
        f"Review: \"{review}\"\n\n"
        "Return the relevant topics as a comma-separated list."
    )

    # tokenize & move to device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )

    # decode and split into individual topics
    decoded = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )
    assigned = [t.strip() for t in decoded.split(",") if t.strip()]
    return assigned
