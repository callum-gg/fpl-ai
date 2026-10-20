import asyncio
import threading


def test_sync_job_runs_off_the_event_loop(db):
    """A long sync job on the loop starves every other cron slot past its misfire grace."""
    from fplai.scheduler import jobs

    seen = {}
    jobs.JOBS["_probe"] = lambda: seen.setdefault("thread", threading.get_ident()) and {}
    try:
        out = asyncio.run(jobs.run_named("_probe"))
    finally:
        del jobs.JOBS["_probe"]
    assert out["status"] == "ok"
    assert seen["thread"] != threading.get_ident()
