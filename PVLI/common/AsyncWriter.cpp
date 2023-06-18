#include "AsyncWriter.h"
#include <iostream>

using namespace std;

void AsyncWriter::run() {
    pipe.start();
    if (senderThread.joinable()) return;
    senderThread = std::thread([this]() {
        try {
            Function f;
            while (pipe.receive(f))
                if (f) sentBytes += f(*socket);
        } catch (const boost::system::system_error& error) {
            cout << "AsyncWriter error (" << error.what() << ")" << std::endl;
        }
    });
}

void AsyncWriter::close(bool closeSocket) {
    pipe.close();
    if (closeSocket && socket) {
        socket->shutdown(boost::asio::ip::tcp::socket::shutdown_send);
        socket->close();
    }
    if (senderThread.joinable())
        senderThread.join();
}