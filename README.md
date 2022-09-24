# ProFatXuanAll.github.io

Notes collection.

## Installation

Only work in Ubuntu.

```sh
# Install rbenv.
sudo apt install rbenv

# Put the output to .bashrc.
rbenv init

# Install ruby 2.7.0
rbenv install 2.7.0
rbenv global 2.7.0

# Install bundler and jekyll.
gem install bundler jekyll

# Clone project.
git clone https://github.com/ProFatXuanAll/ProFatXuanAll.github.io.git
cd ProFatXuanAll.github.io

# Use bundler to install project dependency.
bundle update
```

## Run Dev Server

```sh
bundle exec jekyll serve --livereload --drafts
```

### Production build

```sh
bundle exec jekyll build
```
