import logging
import time

import grpc
import requests

from report_generator import ReportGenerator
from server.rgen.rgen_pb2_grpc import RGenServicer
from server.rgen.rgen_pb2 import Request, Response


class RadiologistServer(RGenServicer):
    def __init__(self, generator: ReportGenerator):
        self.generator = generator

    async def GenerateReport(self, request: Request, context) -> Response:
        logging.debug(f"received request for img @ '{request.link_to_xray}'")
        try:
            report = self.generator.generate_report(request.link_to_xray)
        except requests.RequestException as e:
            logging.error(e)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid link to xray!")
            return Response()
        except Exception as e:
            logging.error(e)
            context.set_code(grpc.StatusCode.INTERNAL)
            return Response()
        return Response(patient_id=request.patient_id, report=report)
