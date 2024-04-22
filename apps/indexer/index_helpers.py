import logging
import elasticsearch as es
from datetime import datetime

from ...utils.constants import ES_INDEX_MAPPING
from . import es_client, INDEX_NAME

def create_index(index_name: str = INDEX_NAME, client: es.Elasticsearch = es_client):
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

def get_last_index_time(index_name: str = INDEX_NAME, client: es.Elasticsearch = es_client) -> datetime:
    """
    Get the last indexed_at timestamp from Elasticsearch.
    """
    try:
        query = {
            "size": 1,
            "sort": [{"indexed_at": {"order": "desc"}}]
        }
        res = client.search(index=index_name, body=query)
        if res['hits']['total']['value'] > 0 and isinstance(res['hits']['hits'][0]['_source']['indexed_at'], datetime):
            return res['hits']['hits'][0]['_source']['indexed_at']
        raise Exception
    except Exception as e:
        logging.error(f"An error occurred while getting the last indexed_at timestamp: {str(e)}")
        return datetime.min