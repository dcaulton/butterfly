import logging
from django.core.management.base import BaseCommand

from chat.controller_kbot import KbotController

logger = logging.getLogger(__name__)



class Command(BaseCommand):
    help = """Calculates embeddings if needed for a particular project.
        Will take the data source file at the location the kbot controller knows for this project,
        then generate embeddings for title and Content at a target location also specified in the kbot controller,
        it will also generate a timestamp file in the target location.
    """

    def add_arguments(self, parser):
        parser.add_argument("-p", "--project", type=str, default=None,
            help="Project that will have its embeddings calculated."
        )

    def handle(self, *args, **options):
        project = options["project"]
        kc = KbotController(project, logger=logger)
        emb_title, emb_content = kc.get_embeddings_title_and_content()
        print(f'embeddings are now available, with {len(emb_title)} records present')
