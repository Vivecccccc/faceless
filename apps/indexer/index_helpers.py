import elasticsearch as es
from ...utils.constants import ES_INDEX_MAPPING

def create_index(index_name: str, client: es.Elasticsearch):
    """
    Create an index in Elasticsearch if it does not exist.
    """
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=ES_INDEX_MAPPING)
    else:
        if client.count(index=index_name)['count'] == 0:
            client.indices.put_mapping(index=index_name, body=ES_INDEX_MAPPING['mappings'])
        else:
            raise Exception(f'Index {index_name} already exists and is not empty.')
    