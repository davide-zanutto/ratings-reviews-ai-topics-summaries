def get_topics(reviews, llm_client, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an helpful customer reviews expert that identifies the main topics discussed in a group of reviews.\n"
                "Use singular words unless a plural form is necessary.\n"                
                "Use only one word. 2 or 3 words can be used only when they are part of a composite word and are better to represent the idea of the topic (e.g.: Ease of use).\n"
                "If you identify a verb as a topic, use the noun version (e.g.: use 'Order' instead of 'Ordering').\n"
                "Generalize the topic word; for example, if you encounter 'Saleswoman' or 'Salesman', abstract it to 'Staff'.\n"
                "Provide the output as a comma-separated list of topics with the first letter capitalized.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Read the following reviews and generate a maximum of 8 topics that are mentioned in the reviews.\n"
                "For each topic that you generate, indicate which reviews mention implicitly or explicitly that topic.\n"
                "ONLY return topics that are mentioned more than once, don't consider topics mentioned only in a couple of reviews.\n"
                "The topic names should be broad and general, for example Quality, Price, etc.\n"
                "The topics could be either nouns that refers to a certain characteristic of the product or spefic features or parts of the product (screws, cookware, etc.)\n"
                "First return all the topics that you identify as a comma-separated list, then for each of them return a few reviews that mention it.\n"
                f"Reviews: {', '.join(reviews)}\n"
                "Topics:"
            ),
        },
    ]

    response = ' '
    
    # Generate the topic word using the language model
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
        n=1,
        stop=None,
    )

    output = response.choices[0].message.content.strip()
    topics = output.split('\n')[0].split(', ')
    # Extract and return the topic words
    return topics