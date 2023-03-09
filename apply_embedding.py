def apply_embedding(x):
    time.sleep(1) # add 1 second delay
    return get_embedding(x, engine=embedding_model)