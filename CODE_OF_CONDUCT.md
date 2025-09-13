# Code of Conduct

## Our Pledge
We as members, contributors, and maintainers of **vtol_rl** pledge to make participation in our project and community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion (or lack thereof), or sexual identity and orientation.

## Our Standards
- 接入规范API（如pytorch dataloader; gymnasium 对RL env return value 的要求；API暴露合理，如有必要需要用pybind11来实现，cpp处理details）
- 最小实现+extra要求的形式 （如VisFly，有些模块无需vision-based）
- 尽可能放弃ROS2,PX4等的依赖；纯并行仿真，以加快迭代速度。



## Scope
This Code applies within all project spaces—GitHub/GitLab, issue trackers, pull requests, discussions, documentation, chats, conferences, and any public spaces when an individual is representing the project or its community.

## Responsibilities
Project maintainers are responsible for clarifying and enforcing this Code of Conduct, and will take appropriate and fair corrective action in response to any behavior they deem inappropriate, threatening, offensive, or harmful.


## Enforcement
All community members are expected to comply with requests to stop unacceptable behavior.  
Maintainers will review all reports and determine appropriate action, which may include:

1. Informal warning  
2. Temporary moderation  
3. Temporary ban from community spaces  
4. Permanent ban from community spaces and/or project contribution


## Confidentiality
We will respect the privacy and security of the reporter and any involved parties.

## Contributor Guidelines (Python-specific)
- Follow PEP 8 and write clear, maintainable Python code.  
- Prefer inclusive language in code, docs, and comments.  
- Be considerate in code review—critique code, not people.  
- Use descriptive commit messages and link issues where relevant.  
- Respect the project’s governance and decision-making processes.

## Attribution
This Code of Conduct draws inspiration from the PSF Code of Conduct and the Contributor Covenant. It has been adapted for **vtol_rl**.

## Contact & Updates
This document was last updated on **2025-09-11**. Suggestions for improvement are welcome via pull request or by emailing **rongerch@outlook.com**.
