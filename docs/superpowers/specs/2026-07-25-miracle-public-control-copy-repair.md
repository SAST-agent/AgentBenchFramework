# Public control-copy reproducibility repair

The historical execution protocol SHA was
`866696fd9e094da85e3f2c04dc5ba20d0500faf8461531a242362323c4efe0b3`.

The tracked protocol is a public, sanitized control copy. Its normalized SHA
identifies only that public copy and does not rewrite historical matrix
provenance. Protocol and roster control hashes normalize UTF-8 line endings to
LF. Strategy, Judge, executable, and archive assets continue to use raw-byte
SHA256 hashing.
