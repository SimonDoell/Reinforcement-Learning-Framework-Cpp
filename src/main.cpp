#include <iostream>
#include <SFML/Graphics.hpp>
#include "Config.hpp"



int main() {
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Boilerplate");
    sf::View view(sf::FloatRect({0, 0}, {WIDTH, HEIGHT}));
    window.setView(view);
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);


    while (window.isOpen()) {
        while (const std::optional<sf::Event> event = window.pollEvent()) {if (event->is<sf::Event::Closed>()) window.close();}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) window.close();

        
        window.clear(sf::Color(31, 31, 31));
        
        window.display();
    }
    

    return 0;
}