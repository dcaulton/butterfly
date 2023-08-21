from django.core.management.base import BaseCommand

from chat.controller_kbot import KbotController
from chat.controller_gpt import GptController


class Command(BaseCommand):
    help = """Sweeps away old jobs to make sure there are not too many.
        Removes the oldest jobs, except those jobs that have an 
        auto_delete_age of "never", until only "keep" number of jobs remain.
        Also attempts to clear all detritus left behind by the job.
    """

    def add_arguments(self, parser):
        parser.add_argument("-k", "--keep", type=int, default=None,
            help="Number of jobs to keep."
        )

    def handle(self, *args, **options):
#        keep = options["keep"]

        kc = KbotController('triboo')
        k_resp = kc.call_all('how about PowerBi access for external clients?')
        print(k_resp)
        gc = GptController()
        g_resp = gc.ask_question('how about PowerBi access for external clients?')
        print(g_resp)


        print('THERE, I just did the thing you want')
