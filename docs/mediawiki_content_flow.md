# How MediaWiki Generates Page Content from SQLite

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. PAGE TABLE                                                       │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │ page_id: 1                                                  │   │
│ │ page_namespace: 0                                           │   │
│ │ page_title: "Main_Page"                                     │   │
│ │ page_latest: 5  ←─────────────┐ (points to current rev)    │   │
│ │ page_len: 996                  │                            │   │
│ └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Follow page_latest
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. REVISION TABLE                                                   │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │ rev_id: 5 ←────────────────────┘                            │   │
│ │ rev_page: 1                                                 │   │
│ │ rev_timestamp: 20240927190340                               │   │
│ │ rev_actor: 3  ──────┐ (who made this edit)                 │   │
│ │ rev_comment_id: 1 ──┼──┐ (edit summary)                    │   │
│ │ rev_parent_id: 4    │  │ (previous revision)               │   │
│ └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
           │                │  │
           │                │  └──→ COMMENT table (comment_text)
           │                └─────→ ACTOR table (actor_name: "~2024-1")
           │
           │ Use rev_id to find content
           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. SLOTS TABLE (Multi-Content Revisions)                           │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │ slot_revision_id: 5 ←─────┘                                 │   │
│ │ slot_role_id: 1 ──────┐ (typically "main" content)          │   │
│ │ slot_content_id: 5 ───┼──┐ (points to actual content)       │   │
│ │ slot_origin: 5        │  │ (revision where content created) │   │
│ └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                           │  │
                           │  │ Join with SLOT_ROLES
                           │  └──→ role_name: "main"
                           │
                           │ Use slot_content_id
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. CONTENT TABLE                                                    │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │ content_id: 5 ←────────┘                                    │   │
│ │ content_size: 996                                           │   │
│ │ content_sha1: [hash]                                        │   │
│ │ content_model: 1 ──────┐ (wikitext, JSON, CSS, etc.)       │   │
│ │ content_address: "tt:5" ──┐ (storage location)             │   │
│ └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                           │   │
                           │   │ Join with CONTENT_MODELS
                           │   └──→ model_name: "wikitext"
                           │
                           │ Parse content_address "tt:5"
                           │ Format: "tt:" = text table
                           ↓              "5" = old_id
┌─────────────────────────────────────────────────────────────────────┐
│ 5. TEXT TABLE (Actual Content Storage)                             │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │ old_id: 5 ←──────────────┘                                  │   │
│ │ old_text: "<strong>MediaWiki has been installed.</strong>  │   │
│ │            <display_map height='300px'...>                  │   │
│ │            Consult the User's Guide..."                     │   │
│ │ old_flags: "utf-8"                                          │   │
│ └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           │ Parse based on content_model & flags
                           ↓
                    ┌──────────────┐
                    │ FINAL OUTPUT │
                    │  (Rendered   │
                    │   HTML)      │
                    └──────────────┘
```

## Step-by-Step Process

### Step 1: Start with PAGE
```sql
SELECT page_id, page_namespace, page_title, page_latest
FROM page
WHERE page_title = 'Main_Page';
```
**Result:** page_id=1, page_latest=5

### Step 2: Get Current REVISION
```sql
SELECT rev_id, rev_timestamp, rev_actor, rev_comment_id
FROM revision
WHERE rev_id = 5;
```
**Result:** Links to actor_id=3, comment_id=1

### Step 3: Find CONTENT via SLOTS
```sql
SELECT slot_content_id, slot_role_id
FROM slots
WHERE slot_revision_id = 5;
```
**Result:** slot_content_id=5, slot_role_id=1 (role="main")

### Step 4: Get CONTENT Metadata
```sql
SELECT content_address, content_model, content_size
FROM content
WHERE content_id = 5;
```
**Result:** content_address="tt:5", content_model=1 (wikitext), size=996

### Step 5: Retrieve Actual TEXT
```sql
SELECT old_text, old_flags
FROM text
WHERE old_id = 5;
```
**Result:** The actual wikitext content

## Key Concepts

### 1. **Content Addressing**
- Format: `tt:X` where X is the `old_id` in the `text` table
- "tt" = "text table" (MediaWiki's blob storage scheme)
- Other possible formats: External storage, compressed blobs, etc.

### 2. **Multi-Slot System (MCR - Multi-Content Revisions)**
- Introduced in MediaWiki 1.32+
- A revision can have multiple "slots" (main content, metadata, etc.)
- Most pages use only the "main" slot
- Enables structured content (e.g., Wikidata items with labels + descriptions)

### 3. **Content Deduplication**
- Multiple revisions can share the same `content_id`
- If you revert to a previous version, `slot_content_id` points to existing content
- `slot_origin` tracks which revision originally created this content
- Saves storage space for reverted edits

### 4. **Content Models**
- **wikitext**: Standard wiki markup (most common)
- **javascript**: User scripts
- **css**: User stylesheets
- **json**: Structured data
- **text**: Plain text

### 5. **Flags in TEXT Table**
- `utf-8`: Standard UTF-8 encoding
- `gzip`: Content is gzip compressed
- `object`: Serialized PHP object
- `external`: Content stored outside database

## Example: Getting Complete Page Content

```sql
-- One query to get everything
SELECT
    p.page_title,
    t.old_text as content,
    cm.model_name as format,
    a.actor_name as last_editor,
    r.rev_timestamp as last_modified
FROM page p
JOIN revision r ON p.page_latest = r.rev_id
JOIN actor a ON r.rev_actor = a.actor_id
JOIN slots s ON r.rev_id = s.slot_revision_id AND s.slot_role_id = 1
JOIN content c ON s.slot_content_id = c.content_id
JOIN content_models cm ON c.content_model = cm.model_id
JOIN text t ON CAST(substr(c.content_address, 4) AS INTEGER) = t.old_id
WHERE p.page_id = 1;
```

## Additional Related Data

### To get page metadata:
- **categorylinks**: Which categories this page belongs to
- **pagelinks**: Internal links FROM this page
- **templatelinks**: Which templates this page uses
- **imagelinks**: Which images this page displays
- **externallinks**: External URLs referenced

### To get edit history:
```sql
-- Get all revisions for a page
SELECT r.rev_id, r.rev_timestamp, a.actor_name, c.comment_text
FROM revision r
JOIN actor a ON r.rev_actor = a.actor_id
JOIN comment c ON r.rev_comment_id = c.comment_id
WHERE r.rev_page = 1
ORDER BY r.rev_timestamp DESC;
```

### To get page protection/restrictions:
```sql
SELECT pr_type, pr_level, pr_expiry
FROM page_restrictions
WHERE pr_page = 1;
```

## Why This Complex Structure?

1. **Efficiency**: Content deduplication saves storage
2. **History**: Complete edit history preserved
3. **Flexibility**: Different content types (wikitext, JSON, CSS)
4. **Multi-content**: Future support for structured content
5. **Scalability**: External blob storage for large wikis
6. **Normalization**: Actors, comments stored once, referenced many times
