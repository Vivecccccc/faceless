import logging
import elasticsearch as es
from datetime import datetime
from typing import List, Optional

from utils.dataclasses import Video
from apps.indexer import es_client, INDEX_NAME
from apps.indexer.index_helpers import check_mapping_consistency

def index_videos(videos: List[Video]):
    is_valid_mapping = False
    try:
        is_valid_mapping = check_mapping_consistency(index_name=INDEX_NAME, raise_for_status=True)
        if not is_valid_mapping:
            raise Exception('Index mapping is not consistent')
    except Exception as e:
        logging.error(f'Error while checking mapping for index {INDEX_NAME}: {e}')
    maybe_retry = []
    if is_valid_mapping:
        for vid in videos:
            try:
                es_client.index(index=INDEX_NAME, document=vid.to_es_obj(), id=vid.id(), op_type='create')
            except es.ConflictError:
                try:
                    es_client.update(index=INDEX_NAME, document=vid.to_es_obj(), id=vid.id())
                except Exception as e:
                    logging.error(f'Error while updating index for previous failure: {e}')
                    maybe_retry.append(vid)
            except Exception as e:
                logging.error(f'Error while creating index for video {vid.id}: {e};')
                maybe_retry.append(vid)
                continue
    return is_valid_mapping, maybe_retry

