import logging
from django.core.management.base import BaseCommand

from chat.redis_session_wrapper import RedisSessionWrapper

logger = logging.getLogger(__name__)



class Command(BaseCommand):
    help = """Clears out redis sessions, optionally for a single project
    """

    def add_arguments(self, parser):
        parser.add_argument("-p", "--project", type=str, default=None,
            help="Project for which to clear sessions."
        )

    def handle(self, *args, **options):
        project = options["project"]
        r = RedisSessionWrapper()
        num_deleted = r.delete_sessions_for_project(project)
        print(f'{num_deleted} sessions have been deleted')


