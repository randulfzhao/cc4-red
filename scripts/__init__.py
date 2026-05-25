# Marker file so `scripts/` can also be imported as a package if needed.
# The individual entry-point scripts under scripts/train/ add the clean cc4
# root to sys.path at runtime, so flat imports such as `from CybORG import ...`
# continue to resolve when a script is invoked directly.
