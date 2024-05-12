import time

from report_generator import ReportGenerator
from server.rgen.rgen_pb2_grpc import RGenServicer
from server.rgen.rgen_pb2 import Request, Response


class RadiologistServer(RGenServicer):
    def __init__(self, generator: ReportGenerator):
        self.generator = generator

    async def GenerateReport(self, request: Request, context) -> Response:
        print(request.patient_id + "\n" + request.link_to_xray)
        time.sleep(6)
        return Response(patient_id=request.patient_id, report="Lungs look very bad. Patient may die tonight.")
