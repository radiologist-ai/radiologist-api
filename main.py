import asyncio
import logging

from grpc import aio

from report_generator import ReportGenerator
from server.conf import GRPC_APP_HOST, GRPC_APP_PORT
from server.rgen import rgen_pb2_grpc
from server.server import RadiologistServer


# Coroutines to be invoked when the event loop is shutting down.
_cleanup_coroutines = []


async def serve():
    path = 'model_best.pth'
    generator = ReportGenerator(path)
    server = aio.server()
    rgen_pb2_grpc.add_RGenServicer_to_server(RadiologistServer(generator=generator), server)

    listen_addr = f"{GRPC_APP_HOST}:{GRPC_APP_PORT}"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()

    async def server_graceful_shutdown():
        logging.info("Starting graceful shutdown...")
        # Shuts down the server with 5 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(5)

    _cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve())
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
