import logging
import elasticsearch as es
from datetime import datetime
from typing import List, Optional

from utils.dataclasses import Video
from apps.indexer import es_client, INDEX_NAME
from apps.indexer.index_helpers import check_mapping_consistency

def index_videos(videos: List[Video], current_indexed_at: Optional[datetime] = None):
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
                if vid.status.is_all_green() and vid.embedding is not None:
                    if current_indexed_at is not None:
                        vid.indexed_at = current_indexed_at # update indexed_at to the latest job
                    try:
                        es_client.update(index=INDEX_NAME, document=vid.to_es_obj(), id=vid.id())
                    except Exception as e:
                        logging.error(f'Error while updating index for previous failure {vid.id}: {e}')
                        maybe_retry.append(vid)
                        continue
                else:
                    logging.error(f'Previous failure {vid.id} not yet resolved')
                    continue
            except Exception as e:
                logging.error(f'Error while creating index for video {vid.id}: {e};')
                maybe_retry.append(vid)
                continue
    return is_valid_mapping, maybe_retry

