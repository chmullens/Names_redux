# Names_redux
Rebuild of earlier Names project with better structure. I often find myself wishing I had this conveniently available, just so I could answer a question for myself about when a name was most popular. 

Basic outline, build:

- Import name file from social security site (this specific address has worked for at least five years so far): https://www.ssa.gov/oact/babynames/names.zip
- Import actuarial data I generated during the previous project. This was basically its own little research project to boil down the numbers nicely, it's under 1MB total. 
- Preparse names. My main goal here is to precalculate the math more efficiently, so that it can work conveniently as a webpage. 

