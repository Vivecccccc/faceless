import logging
from typing import List

from ..utils.dataclasses import Video
from ..apps.indexer import es_client
from ..apps.indexer.index_helpers import create_index

def index_videos(videos: List[Video]):
    try:
        index_created = create_index(exists_ok=False)
    except Exception as e:
        logging.error(f'Error while checking index: {e}')
    for vid in videos:
        if vid.status.is_all_green() and vid.embedding is not None:
            try:
                resp = es_client.index(index=index_created, document=vid.to_es_body(), id=vid.id(), op_type='create')
                if resp['result'] != 'created':
                    raise Exception(f'detailed error: {resp}')
            except Exception as e:
                logging.error(f'Error while indexing video {vid.id}: {e};')

