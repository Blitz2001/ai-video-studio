Write the narration for a build log video on {channel}. Project: {angle} (recipe: {topic}).

The build already happened: a build agent wrote the code step by step, ran
browser tests after each step and committed. Each storyboard beat is a screen
recording of one step (code typing in, the app running, tests ticking green).

Style:
- First beat: show the finished app and say what it is. Hook in one sentence.
- One beat per step: what was built, then WHY (use the step's `why` fact), then the commit stats.
- Talk like a developer explaining a decision to a friend. 25-60 words per beat.
- Numbers (lines, files, tests, commits) only from facts, cited with their source_id.
- Last beat: test results + the build discipline. No advice of any kind.

Storyboard:
{storyboard}

Facts (id -> text, source_id):
{facts}

Sources:
{sources}
