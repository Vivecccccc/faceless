import logging
import elasticsearch as es

from datetime import datetime
from typing import List, Optional
from collections import defaultdict

from utils.dataclasses import Video, Duplicates, Returns
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
                    es_client.index(index=INDEX_NAME, document=vid.to_es_obj(), id=vid.id(), op_type='index')
                except Exception as e:
                    logging.error(f'Error while updating index for previous failure: {e}')
                    maybe_retry.append(vid)
            except Exception as e:
                logging.error(f'Error while creating index for video {vid.id}: {e};')
                maybe_retry.append(vid)
                continue
    return is_valid_mapping, maybe_retry

def get_top_k(k: int, num_candidates: int, threshold: float, videos: List[Video]) -> List[Returns]:
    # for dot_product, actual_similarity = _score * 2 - 1
    results = []
    for video in videos:
        duplicates = []
        if not video.status.is_all_green() or not isinstance(video.embedding, List):
            results.append(Returns(status=False, application_id=video.metadata.application_id, duplicates=duplicates))
            continue
        query_body = {
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": video.embedding,
                "k": k,
                "num_candidates": num_candidates,
                "similarity": threshold,
                "filter": {
                    "bool": {
                        "must_not": {
                            "term": {
                                "_id": video.id
                            }
                        }
                    }
                }
            }
        }
        try:
            response = es_client.search(index=INDEX_NAME, body=query_body)
        except Exception as e:
            logging.error(f'Error while fetching top {k} similar videos for {video.id}: {e}')
            results.append(Returns(status=False, application_id=video.metadata.application_id, duplicates=duplicates))
            continue
        hits = response['hits']['hits']
        for hit in hits:
            if hit['_id'] == video.id:
                continue
            candidate = hit['_source']['metadata']['application_id']
            similarity = hit['_score'] * 2 - 1
            duplicates.append(Duplicates(application_id=candidate, score=similarity))
        results.append(Returns(status=True, application_id=video.metadata.application_id, duplicates=duplicates))
    return results