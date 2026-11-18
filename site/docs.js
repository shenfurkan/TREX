(function () {
  'use strict';

  var allowed = new Set(Array.prototype.map.call(
    document.querySelectorAll('[data-page]'),
    function (link) { return link.getAttribute('data-page'); }
  ));
  allowed.add('docs/index.md');

  var params = new URLSearchParams(window.location.search);
  var page = params.get('page') || 'docs/README.md';
  if (!allowed.has(page)) page = 'docs/README.md';

  var pageTypes = {
    'README.md': 'Overview',
    'concepts.md': 'Explanation',
    'tutorial.md': 'Tutorial',
    'how-trex-works.md': 'Explanation',
    'installation.md': 'How-to',
    'inputs.md': 'Reference',
    'outputs.md': 'Reference',
    'api.md': 'Reference',
    'prepared-runs.md': 'How-to',
    'from-tic.md': 'How-to',
    'from-scene.md': 'How-to',
    'background-population.md': 'Explanation',
    'scenarios.md': 'Reference',
    'modules.md': 'Reference',
    'glossary.md': 'Reference',
    'faq.md': 'How-to',
    'differences-from-triceratops.md': 'Explanation'
  };

  var article = document.getElementById('doc');
  var raw = document.getElementById('raw');
  var sidebar = document.querySelector('.sidebar');
  var search = document.getElementById('search');
  var searchStatus = document.getElementById('search-status');
  var menuToggle = document.querySelector('.menu-toggle');
  raw.href = page;

  document.querySelectorAll('[data-page]').forEach(function (link) {
    if (link.getAttribute('data-page') === page) {
      link.setAttribute('aria-current', 'page');
    }
  });

  setupSearch();
  setupMenu();

  fetch(page, { headers: { 'Accept': 'text/plain' } })
    .then(function (response) {
      if (!response.ok) throw new Error(response.status + ' ' + response.statusText);
      return response.text();
    })
    .then(function (markdown) {
      article.innerHTML = render(markdown);
      rewriteLinks(article);
      buildPageMeta();
      wrapTables();
      addCopyButtons();
      buildToc();
      trackReading();
      buildPager();
      var heading = article.querySelector('h1');
      if (heading) document.title = heading.textContent + ' — TREX documentation';
      if (window.location.hash) {
        var target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
        if (target) target.scrollIntoView();
      }
    })
    .catch(function (error) {
      article.innerHTML = '<p class="load-error">Could not load <code>' +
        escapeHtml(page) + '</code>: ' + escapeHtml(error.message) +
        '. Read the <a href="' + page + '">raw Markdown</a> instead.</p>';
    });

  function buildToc() {
    var toc = document.getElementById('toc');
    var headings = article.querySelectorAll('h2, h3');
    if (headings.length < 2) return;
    var html = '<p class="group">On this page</p>';
    Array.prototype.forEach.call(headings, function (heading) {
      html += '<a href="#' + heading.id + '" data-level="' + heading.tagName.slice(1) + '">' +
        escapeHtml(heading.textContent) + '</a>';
    });
    toc.innerHTML = html;
    toc.hidden = false;
  }

  function buildPager() {
    var pager = document.getElementById('pager');
    var links = Array.prototype.slice.call(document.querySelectorAll('.sidebar [data-page]'));
    var index = links.findIndex(function (link) {
      return link.getAttribute('data-page') === page;
    });
    if (index < 0) return;
    var previous = links[index - 1];
    var next = links[index + 1];
    if (!previous && !next) return;
    var html = previous
      ? '<a class="prev" href="' + previous.getAttribute('href') + '"><small>Previous</small>' +
        escapeHtml(previous.textContent) + '</a>'
      : '<span></span>';
    html += next
      ? '<a class="next" href="' + next.getAttribute('href') + '"><small>Next</small>' +
        escapeHtml(next.textContent) + '</a>'
      : '<span></span>';
    pager.innerHTML = html;
    pager.hidden = false;
  }

  function setMenu(open) {
    if (!sidebar || !menuToggle) return;
    sidebar.classList.toggle('nav-open', open);
    menuToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    menuToggle.textContent = open ? 'Close' : 'Menu';
  }

  function setupMenu() {
    if (!menuToggle) return;
    menuToggle.addEventListener('click', function () {
      setMenu(!sidebar.classList.contains('nav-open'));
    });
  }

  function setupSearch() {
    if (!search) return;
    search.addEventListener('input', applySearch);
    search.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') {
        search.value = '';
        applySearch();
        return;
      }
      if (event.key !== 'Enter' || !search.value.trim()) return;
      var first = document.querySelector('#doc-nav a:not([hidden])');
      if (first) window.location.href = first.getAttribute('href');
    });
  }

  function applySearch() {
    var query = search.value.trim().toLowerCase();
    var groups = [];
    var current = null;
    Array.prototype.forEach.call(document.querySelectorAll('#doc-nav > *'), function (node) {
      if (node.classList.contains('group')) {
        current = { heading: node, links: [] };
        groups.push(current);
      } else if (node.tagName === 'A' && current) {
        current.links.push(node);
      }
    });
    var matches = 0;
    groups.forEach(function (group) {
      var shown = 0;
      group.links.forEach(function (link) {
        var visible = !query || link.textContent.toLowerCase().indexOf(query) !== -1;
        link.hidden = !visible;
        if (visible) shown += 1;
      });
      group.heading.hidden = Boolean(query) && shown === 0;
      matches += shown;
    });
    searchStatus.textContent = query
      ? (matches ? matches + ' page' + (matches === 1 ? '' : 's') : 'No matching pages')
      : '';
    if (query) setMenu(true);
  }

  function buildPageMeta() {
    var link = document.querySelector('.sidebar [data-page][aria-current="page"]');
    var rest = document.getElementById('crumb-rest');
    var group = '';
    if (link) {
      var node = link.previousElementSibling;
      while (node && !node.classList.contains('group')) node = node.previousElementSibling;
      if (node) group = node.textContent.trim();
    }
    var heading = article.querySelector('h1');
    var label = link ? link.textContent.trim() : (heading ? heading.textContent : '');
    var html = '';
    if (group) html += ' <span aria-hidden="true">/</span> <span>' + escapeHtml(group) + '</span>';
    if (label) html += ' <span aria-hidden="true">/</span> <span aria-current="page">' + escapeHtml(label) + '</span>';
    rest.innerHTML = html;

    var type = pageTypes[page.split('/').pop()];
    if (type) {
      var tag = document.createElement('p');
      tag.className = 'page-type';
      tag.textContent = type;
      article.insertBefore(tag, article.firstChild);
    }
  }

  function wrapTables() {
    article.querySelectorAll('table').forEach(function (table) {
      var wrapper = document.createElement('div');
      wrapper.className = 'table-scroll';
      wrapper.tabIndex = 0;
      wrapper.setAttribute('role', 'region');
      var header = table.querySelector('th');
      wrapper.setAttribute('aria-label', header ? header.textContent + ' table' : 'Table');
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  }

  function addCopyButtons() {
    article.querySelectorAll('pre').forEach(function (pre) {
      var wrapper = document.createElement('div');
      wrapper.className = 'code-block';
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'copy';
      button.textContent = 'Copy';
      button.addEventListener('click', function () {
        copyText(pre.textContent, button);
      });
      wrapper.appendChild(button);
    });
  }

  function copyText(text, button) {
    function done(ok) {
      button.textContent = ok ? 'Copied' : 'Copy failed';
      window.setTimeout(function () { button.textContent = 'Copy'; }, 2000);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        done(true);
      }, function () {
        done(fallbackCopy(text));
      });
      return;
    }
    done(fallbackCopy(text));
  }

  function fallbackCopy(text) {
    var area = document.createElement('textarea');
    area.className = 'copy-source';
    area.value = text;
    area.setAttribute('readonly', '');
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (error) {
      ok = false;
    }
    area.remove();
    return ok;
  }

  function trackReading() {
    var links = Array.prototype.slice.call(document.querySelectorAll('#toc a'));
    if (!links.length || !('IntersectionObserver' in window)) return;
    var byId = {};
    links.forEach(function (link) {
      byId[link.getAttribute('href').slice(1)] = link;
    });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var link = byId[entry.target.id];
        if (!link) return;
        links.forEach(function (item) { item.removeAttribute('aria-current'); });
        link.setAttribute('aria-current', 'true');
      });
    }, { rootMargin: '0px 0px -70% 0px' });
    article.querySelectorAll('h2, h3').forEach(function (heading) {
      observer.observe(heading);
    });
  }

  function render(markdown) {
    var lines = markdown.replace(/\r\n/g, '\n').split('\n');
    var out = [];
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      if (!line.trim()) { i += 1; continue; }

      if (/^```/.test(line)) {
        var language = line.slice(3).trim();
        i += 1;
        var code = [];
        while (i < lines.length && !/^```/.test(lines[i])) {
          code.push(lines[i]);
          i += 1;
        }
        i += 1;
        out.push('<pre><code' + (language ? ' class="language-' + escapeHtml(language) + '"' : '') +
          '>' + escapeHtml(code.join('\n')) + '</code></pre>');
        continue;
      }

      var heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        var level = heading[1].length;
        var text = heading[2].trim();
        out.push('<h' + level + ' id="' + slug(text) + '">' + inline(text) + '</h' + level + '>');
        i += 1;
        continue;
      }

      if (line.trim().charAt(0) === '|' && i + 1 < lines.length &&
          /^\|?[\s:|-]+\|?$/.test(lines[i + 1].trim())) {
        var table = renderTable(lines, i);
        out.push(table.html);
        i = table.next;
        continue;
      }

      if (/^\s*[-*]\s+/.test(line)) {
        var unordered = renderList(lines, i, false);
        out.push(unordered.html);
        i = unordered.next;
        continue;
      }

      if (/^\s*\d+\.\s+/.test(line)) {
        var ordered = renderList(lines, i, true);
        out.push(ordered.html);
        i = ordered.next;
        continue;
      }

      if (/^>\s?/.test(line)) {
        var quoted = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) {
          quoted.push(lines[i].replace(/^>\s?/, ''));
          i += 1;
        }
        out.push('<blockquote><p>' + inline(quoted.join(' ')) + '</p></blockquote>');
        continue;
      }

      if (/^---+$/.test(line.trim())) {
        out.push('<hr>');
        i += 1;
        continue;
      }

      var paragraph = [];
      while (i < lines.length && lines[i].trim() && !isBlockStart(lines, i)) {
        paragraph.push(lines[i].trim());
        i += 1;
      }
      out.push('<p>' + inline(paragraph.join(' ')) + '</p>');
    }

    return out.join('\n');
  }

  function isBlockStart(lines, i) {
    var line = lines[i];
    if (/^#{1,6}\s+/.test(line) || /^```/.test(line) || /^>\s?/.test(line) ||
        /^\s*[-*]\s+/.test(line) || /^\s*\d+\.\s+/.test(line) ||
        /^---+$/.test(line.trim())) {
      return true;
    }
    return line.trim().charAt(0) === '|' && i + 1 < lines.length &&
      /^\|?[\s:|-]+\|?$/.test(lines[i + 1].trim());
  }

  function renderTable(lines, start) {
    var headers = cells(lines[start]);
    var i = start + 2;
    var rows = [];
    while (i < lines.length && lines[i].trim().charAt(0) === '|') {
      rows.push(cells(lines[i]));
      i += 1;
    }
    var html = '<table><thead><tr>' + headers.map(function (cell) {
      return '<th>' + inline(cell) + '</th>';
    }).join('') + '</tr></thead><tbody>';
    rows.forEach(function (row) {
      html += '<tr>' + row.map(function (cell) {
        return '<td>' + inline(cell) + '</td>';
      }).join('') + '</tr>';
    });
    return { html: html + '</tbody></table>', next: i };
  }

  function cells(line) {
    return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(function (cell) {
      return cell.trim();
    });
  }

  function renderList(lines, start, ordered) {
    var pattern = ordered ? /^\s*\d+\.\s+/ : /^\s*[-*]\s+/;
    var items = [];
    var i = start;
    while (i < lines.length) {
      var match = lines[i].match(pattern);
      if (match) {
        items.push(lines[i].replace(pattern, ''));
        i += 1;
        continue;
      }
      if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
        items[items.length - 1] += ' ' + lines[i].trim();
        i += 1;
        continue;
      }
      break;
    }
    var tag = ordered ? 'ol' : 'ul';
    return {
      html: '<' + tag + '>' + items.map(function (item) {
        return '<li>' + inline(item) + '</li>';
      }).join('') + '</' + tag + '>',
      next: i
    };
  }

  function inline(text) {
    var value = text.replace(/<((?:https?:\/\/|doi:)[^>\s]+)>/g, function (_, url) {
      return '[' + url + '](' + url + ')';
    });
    value = escapeHtml(value);
    var code = [];
    value = value.replace(/`([^`]+)`/g, function (_, body) {
      code.push(body);
      return '\u0000' + (code.length - 1) + '\u0000';
    });
    value = value.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    value = value.replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');
    value = value.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_, label, href) {
      return '<a href="' + href.replace(/"/g, '&quot;') + '">' + label + '</a>';
    });
    value = value.replace(/\u0000(\d+)\u0000/g, function (_, index) {
      return '<code>' + code[Number(index)] + '</code>';
    });
    return value;
  }

  function slug(text) {
    return text.replace(/`/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .trim()
      .replace(/\s+/g, '-') || 'section';
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function rewriteLinks(root) {
    root.querySelectorAll('a').forEach(function (link) {
      var href = link.getAttribute('href');
      if (!href) return;
      if (/^https?:\/\//.test(href) || href.indexOf('mailto:') === 0) {
        link.target = '_blank';
        link.rel = 'noopener';
        return;
      }
      if (href.charAt(0) === '#') return;
      var match = href.match(/^\.?\/?([A-Za-z0-9._-]+\.md)(#.*)?$/);
      if (match) {
        link.setAttribute('href', '?page=' + encodeURIComponent('docs/' + match[1]) + (match[2] || ''));
      }
    });
  }
})();
