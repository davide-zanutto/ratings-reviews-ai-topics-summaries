def get_reviews_summary(reviews, llm_client, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a skilled summarizer specializing in customer feedback analysis.\n"
                "Your role is to identify and concisely summarize the main aspects, sentiments, and frequently mentioned points in customer reviews.\n"
                "The reviews provided are related to an IKEA product and may discuss various aspects such as the quality, ease of assemble or its value for money.\n" 
                "The summary you generate will be used by customers to understand in a few words what other customers think about the product.\n"           
                "Provide the output as a short text summary with no more than 70 words. Do not exceed this limit.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Please read carefully the following customer reviews and generate a summary of the main aspects that customers are discussing.\n"
                "The summary should be as concise as possible, only reporting the main aspects.\n"
               """Touches all main aspects and topics mentioned in the reviews
                    - Most mentioned aspects should be given priority by writing them first
                    - For each aspect: 
                        - If the overall sentiment for that aspect is positive, mention what customer liked about it
                        - If the overall sentiment for that aspect is negative, mention what customer disliked about it
                        - If the overall sentiment for that aspect is neutral, briefly mention both positive and negative aspects
                    - The summary should follow the general sentiment of the customers
                        - If the overall sentiment is positive (high rating), the summary should begin listing the positive characteristics of the product
                        - If the overall sentiment is negative (low rating), the summary should begin listing the negative characteristics of the product
                        - If the overall sentiment is neutral (median rating), the summary should be objective and list the conflicting opinions on aspects
                    - The summary should be written in a way that is easy to understand by customers and does not include technical jargon or complicated terms
                    - The summary should not include any information that is reported by a negligible number of reviews or not mentioned at all\n"""
                    "Do not mention the sentiment in the summary, use 'Customers like or customers dislike' instead.\n"
                "I will provide you with the reviews and you will generate the summary.\n"
                f"Reviews: {reviews}\n"
                "Summary:\n"
            ),
        },
    ]

    # Generate the topic word using the language model
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.5,
        n=1,
        stop=None,
    )

    # Extract and return the topic word
    return response.choices[0].message.content.strip()