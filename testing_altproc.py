# There has to be a more efficient way to generate the names than the way I'm doing things. Right now, I'm calculating
# the number alive in every year, for every birth year, for every name. That must be something I can preprocess to be
# more efficient.
# What do I need that for right now?
#   - I look at the target span, age, and sex, then find the names that are most characteristic for that span. This
#     doesn't actually change much at all with deaths! If I did a births-only version, that would drastically (REALLY
#     drastically) chop down the matrix size. It limits the cool graphs I can make, but might be worth it. I'll have to
#     look and see if my math gets angry.
# What might get lost?
#

# Right now, it's very flexible; you can give it any year span, and it'll give you the most-characteristic options. What
# would a trimmed version look like? Have considered getting any five- or ten-year span; that would allow me to parse
# down the options a good bit

# Option 1: Rank all names



