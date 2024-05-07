import logging
import elasticsearch as es
from typing import Optional
from datetime import datetime, timedelta

from utils.constants import ES_INDEX_MAPPING
from utils.exceptions import ESIndexMappingException, ESRecordsException
from . import es_client, INDEX_NAME

def create_or_check_index(index_name: str = INDEX_NAME, client: es.Elasticsearch = es_client, exists_ok: bool = True) -> Optional[str]:
    """
    Create an index in Elasticsearch if it does not exist.
    """
    if not client.indices.exists(index=index_name):
        try:
            client.indices.create(index=index_name, body=ES_INDEX_MAPPING)
            return index_name
        except Exception as e:
            raise ESIndexMappingException(f"An error occurred while creating index {index_name}: {e}")
    else:
        if client.count(index=index_name)['count'] == 0:
            try:
                client.indices.put_mapping(index=index_name, body=ES_INDEX_MAPPING['mappings'])
                return index_name
            except Exception as e:
                raise ESIndexMappingException(f"An error occurred while updating mapping for index {index_name}: {e}")
        else:
            if not exists_ok:
                raise ESIndexMappingException(f'Index {index_name} already exists and is not empty. Set exists_ok=True to bypass this error')
            else:
                is_same_mapping = check_mapping_consistency(index_name=index_name, client=client, mapping=ES_INDEX_MAPPING)
                return index_name if is_same_mapping else None

def check_mapping_consistency(index_name: str = INDEX_NAME,
                              client: es.Elasticsearch = es_client,
                              mapping: dict = ES_INDEX_MAPPING,
                              raise_for_status: bool = False) -> bool:
    """
    Check if the mapping of an index in Elasticsearch is consistent with the expected mapping.
    """
    try:
        current_mapping = client.indices.get_mapping(index=index_name)[index_name]
        return current_mapping == mapping
    except Exception as e:
        if raise_for_status:
            raise e
        logging.error(ESIndexMappingException(f"An error occurred while checking mapping consistency for index {index_name}: {e}"))
    return False

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
        raise ESRecordsException(f'No records found in index {INDEX_NAME}')
    except Exception as e:
        logging.warning(f"An error occurred while getting the last indexed_at timestamp: {e}")
        return datetime.now() - timedelta(days=1)