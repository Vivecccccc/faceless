import os

META_CONSTANTS = {
    'VIDEO_EXTENSION': '.mp4',
    'TEMP_VIDEO_STORAGE': '.cache/videos/',
    'TEMP_IMAGE_STORAGE': '.cache/images/',
}

S3_CONSTANTS = {
    'AWS_ACCESS_KEY_ID': os.environ.get('AWS_ACCESS_KEY_ID'),
    'AWS_SECRET_ACCESS_KEY': os.environ.get('AWS_SECRET_ACCESS_KEY'),
    'ENDPOINT_URL': os.environ.get('AWS_ENDPOINT_URL'),
    'BUCKET_NAME': os.environ.get('AWS_BUCKET_NAME'),
    'REGION_NAME': os.environ.get('AWS_REGION_NAME'),
}

PORTAL_CONSTANTS = {
    'ENDPOINT_URL': os.environ.get('PORTAL_ENDPOINT_URL'),
}

ES_CONSTANTS = {
    'ENDPOINT_URL': os.environ.get('ES_ENDPOINT_URL'),
    'BASIC_AUTH': (os.environ.get('ES_USERNAME'), os.environ.get('ES_PASSWORD')),
    'INDEX_NAME': 'datasets-videos',
    'CA_CERTS': os.environ.get('ES_CA_CERTS'),
}

ES_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "indexed_at": {"type": "date"},
            "metadata": {
                "properties": {
                    "application_id": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text"
                            }
                        }
                    },
                    "s3_file_key": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text"
                            }
                        }
                    },
                    "created_at": {"type": "date"},
                }
            },
            "status": {
                "properties": {
                    "fetched": {"type": "integer"},
                    "captured": {"type": "integer"},
                    "peated": {"type": "integer"},
                }
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 512,
                "similarity": "dot_product",
                "index": True
            },
            "feedback": {
                "properties": {
                    "flag": {"type": "boolean"},
                    "duplicates": {
                        "type": "nested",
                        "properties": {
                            "id": {"type": "keyword"},
                            "score": {"type": "float"}
                        }
                    }
                }
            }
        }
    }
}

DETECTOR_CONSTANTS = {
    'CROP_SIZE': 112,
    'SCALE': 1.0,
    'NUM_FRAMES': 60,
}

RECOGNIZER_CONSTANTS = {
    'BATCH_SIZE': 128,
    'CKPT_PATH': None,
    'BACKBONE_CKPT_PATH': 'static/backbone.pth',
}

for k, v in META_CONSTANTS.items():
    if k in ['TEMP_VIDEO_STORAGE', 'TEMP_IMAGE_STORAGE']:
        if not os.path.exists(v):
            os.makedirs(v)