class DataMismatchException(Exception):
    def __init__(self, message):
        super().__init__(message)

class ESIndexMappingException(Exception):
    def __init__(self, message):
        super().__init__(message)

class ESRecordsException(Exception):
    def __init__(self, message):
        super().__init__(message)

class S3FetchException(Exception):
    def __init__(self, message):
        super().__init__(message)

class PortalConnectionException(Exception):
    def __init__(self, message):
        super().__init__(message)

class VideoFileInvalidException(Exception):
    def __init__(self, message):
        super().__init__(message)

class H5FileInvalidException(Exception):
    def __init__(self, message):
        super().__init__(message)

class ModelInitException(Exception):
    def __init__(self, message):
        super().__init__(message)

class StageFetchException(Exception):
    def __init__(self, message):
        super().__init__(message)

class StageCaptureException(Exception):
    def __init__(self, message):
        super().__init__(message)

class StagePeatException(Exception):
    def __init__(self, message):
        super().__init__(message)

class StageIndexException(Exception):
    def __init__(self, message):
        super().__init__(message)