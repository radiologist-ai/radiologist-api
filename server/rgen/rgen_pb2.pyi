from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Request(_message.Message):
    __slots__ = ("patient_id", "link_to_xray")
    PATIENT_ID_FIELD_NUMBER: _ClassVar[int]
    LINK_TO_XRAY_FIELD_NUMBER: _ClassVar[int]
    patient_id: str
    link_to_xray: str
    def __init__(self, patient_id: _Optional[str] = ..., link_to_xray: _Optional[str] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ("patient_id", "report")
    PATIENT_ID_FIELD_NUMBER: _ClassVar[int]
    REPORT_FIELD_NUMBER: _ClassVar[int]
    patient_id: str
    report: str
    def __init__(self, patient_id: _Optional[str] = ..., report: _Optional[str] = ...) -> None: ...
