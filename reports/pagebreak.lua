-- Pandoc filter: insert a Word page break before every level-1 heading.
-- Used by `make report` so page breaks live in the build, not in report.md
-- (which stays clean GitHub-flavoured markdown). The title page (title.md) is
-- concatenated before report.md, so the first break separates cover from body.
function Header(el)
  if el.level == 1 then
    local pb = pandoc.RawBlock('openxml',
      '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    return { pb, el }
  end
end
