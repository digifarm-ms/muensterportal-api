# Muenster4You

FastAPI application with search functionality.

## Weekly wiki ingestion

The MediaWiki content is mirrored into a LanceDB store on the server every
Sunday at 03:00 UTC by the workflow `.github/workflows/ingest.yml`. It can
also be triggered manually via **Actions → Weekly Wiki Ingestion → Run
workflow**.

### What it does

The workflow SSHes into the server (using the same `SERVER_HOST`,
`SERVER_USER`, `SERVER_SSH_KEY` secrets as the deploy workflow) and runs
`python -m muenster4you.ingest` against a fresh copy of the live MediaWiki
SQLite database. Output goes to `~/lancedb` (sibling of
`~/muensterportal-api`).

### Server-side setup (one-time)

The MediaWiki instance runs as the `tomwiki` user; its home directory is
`0700`, so `deploy` cannot read the wiki SQLite directly. Read access is
granted via POSIX ACLs:

```bash
sudo apt-get install -y acl

# Traverse permission down to the cache dir
sudo setfacl -m u:deploy:rx \
  /home/tomwiki \
  /home/tomwiki/mediawiki-1.44.0 \
  /home/tomwiki/mediawiki-1.44.0/cache \
  /home/tomwiki/mediawiki-1.44.0/cache/sqlite

# Read permission on the SQLite file itself
sudo setfacl -m u:deploy:r \
  /home/tomwiki/mediawiki-1.44.0/cache/sqlite/my_wiki.sqlite
```

`uv` must be installed for the `deploy` user (one-time):

```bash
sudo -u deploy bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
```

### When MediaWiki is upgraded

The MediaWiki version is hardcoded in the SQLite source path
(`/home/tomwiki/mediawiki-<version>/cache/sqlite/my_wiki.sqlite`). After an
upgrade:

1. Update `SQLITE_SRC` in `.github/workflows/ingest.yml` to the new path.
2. Re-run the `setfacl` commands above against the new path.
