"""foresight.envs package.

Deliberately empty at import time. The legacy single-agent
`instinct_gridworld` env still works but it's the only one that pulls in
`gymnasium`, so we don't import it here — that way the no_free_signal server (which
only needs the unified world) loads cleanly even if gymnasium isn't
installed. Import the legacy env explicitly when you want it:

    from foresight.envs.instinct_gridworld import InstinctGridworldEnv
"""
