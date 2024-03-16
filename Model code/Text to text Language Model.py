pip install cohere

pip install annoy

import cohere
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from annoy import AnnoyIndex
import numpy as np
import pandas as pd

#Sample prompt below
question = "Which Sydney beach should I visit?"


#Training data
text = """

Sydney is world famous for beautiful beaches. These beaches offer different vibes and attractions, from bustling crowds and great surfing conditions to more tranquil and family-friendly environments. 

Bondi Beach: Bondi is perhaps the most famous beach in Sydney, if not Australia. It's known for its golden sands, vibrant atmosphere, and excellent surfing conditions. The Bondi to Coogee coastal walk offers stunning views of the coastline.

Manly Beach: Easily accessible by a scenic ferry ride from Circular Quay, Manly Beach is known for its laid-back atmosphere and family-friendly environment. The beach is great for swimming, surfing, and various water sports.

Cronulla Beach: Located in the Sutherland Shire, Cronulla offers a more relaxed atmosphere compared to some of the busier city beaches. It's a great spot for swimming, picnicking, and enjoying a range of seaside cafes and restaurants.

Bronte Beach: Situated between Bondi and Coogee, Bronte Beach is popular among both locals and visitors. It's a smaller, quieter beach with a beautiful park area and a natural rock pool that's ideal for swimming.

Tamarama Beach: Also known as "Glamarama" due to its popularity among the fashion-conscious crowd, Tamarama Beach is a smaller and more secluded option. It's surrounded by rocky cliffs and offers strong surf, making it a favorite among experienced surfers.

"""

# Split into a list of paragraphs
texts = text.split('\n\n')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts if t])


# Using the API of Embed model from Cohere
co = cohere.Client('PbrfoQaDsYWDOIcqctLKJjKUakayNAJudibQafl8')


# Get the embeddings
response = co.embed(

    texts=texts.tolist(),

).embeddings

# Check the dimensions of the embeddings
embeds = np.array(response)

# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')

# Add all the vectors to the search index
for i in range(len(embeds)):

    search_index.add_item(i, embeds[i])

search_index.build(10) # 10 trees
search_index.save('test.ann')


def search_text(query):

    # Get the query's embedding for the model to recognize the prompt and compare the prompt with the vectors in embedding space
    query_embed = co.embed(texts=[query]).embeddings

    

    # Retrieve the nearest neighbors for finding similiar texts
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],

                                                    10,

                                                  include_distances=True)



    search_results = texts[similar_item_ids[0]]

    

    return search_results

results = search_text(question)

print(results[0])

def ask_llm(question, num_generations=1):

    # Search the text archive

    results = search_text(question)


    # Get the top result

    context = results[0]


    # Prepare the prompt

    prompt = f"""

    More information about Australian beaches at australia.com: 

    {context}

    Question: {question}

    

    Extract the answer of the question from the text provided. 

    If the text doesn't contain the answer, 

    reply that the answer is not available."""




    prediction = co.generate(

        prompt=prompt,

        max_tokens=70,

        model="command-nightly",

        temperature=0.5,

        num_generations=num_generations

    )
    return prediction.generations


results = ask_llm(question,)

print(results[0])


question = "Which Sydney beach is for family?"
results = ask_llm(question,)
print(results[0])

question = "Sydney is considered as the family-friendly environment beaches, have you got it now?"
results = ask_llm(question,)
print(results[0])

question = "Which Sydney beach has a rock pool?"
results = ask_llm(question,)
print(results[0])
