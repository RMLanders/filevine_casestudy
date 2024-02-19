import marqo

url = "http://localhost:8882"
mq = marqo.Client(url=url)

doc = mq.index("whole").get_document(
    document_id="00295e5d-1860-45dd-96a9-e410e13fe233",
    expose_facets=True
)

