require 'nokogiri'

module AgentDirectory
  class Generator < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      # Use Jekyll's file list so excluded and unpublished pages stay out.
      # Only first-level subdirectories (/agent/<name>/), not pages nested deeper.
      entries = (site.static_files + site.pages).select do |entry|
        entry.url.match?(%r{\A/agent/[^/]+/(?:index\.html)?\z}) && entry.write?
      end

      site.data['agent_directory'] = entries.sort_by(&:url).map do |entry|
        source = entry.is_a?(Jekyll::Page) ? entry.content : File.read(entry.path, encoding: 'UTF-8')
        document = Nokogiri::HTML(source)
        url = entry.url.sub(/index\.html\z/, '')
        title = entry.data['title'] || document.at_css('title')&.text || File.basename(url)
        description = entry.data['description'] || document.at_css('meta[name="description"]')&.[]('content')

        {
          'url' => url,
          'title' => title.to_s.gsub(/\s+/, ' ').strip,
          'description' => description.to_s.gsub(/\s+/, ' ').strip
        }
      end
    end
  end
end
