# Báo Cáo Nghiên Cứu Chuyên Sâu: Kiến Trúc Neural Network và Vấn Đề Hành Vi Lặp Vòng Trong Học Tăng Cường Tại Môi Trường Snake và Maze

Học tăng cường (Reinforcement Learning - RL) đã khẳng định vị thế là một trong những phương pháp tiếp cận mạnh mẽ nhất để giải quyết các bài toán ra quyết định tuần tự, cho phép các tác tử (agents) tự động khám phá và thiết lập các chiến lược tối ưu thông qua quá trình tương tác liên tục với môi trường. Khác với học có giám sát truyền thống dựa trên các bộ dữ liệu được gán nhãn sẵn, học tăng cường mô phỏng quá trình học tập tự nhiên của sinh vật, nơi tác tử thực hiện hành động, nhận phản hồi dưới dạng phần thưởng (reward) hoặc hình phạt (penalty), và điều chỉnh chính sách của mình để tối đa hóa tổng phần thưởng chiết khấu trong tương lai. 

Mặc dù các thuật toán Học Tăng Cường Sâu (Deep Reinforcement Learning - DRL) như Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), hay Asynchronous Advantage Actor-Critic (A3C) đã vượt qua khả năng của con người trong vô số bài toán phức tạp, chúng vẫn bộc lộ những điểm yếu chí mạng khi đối mặt với các môi trường có tính quan sát một phần và phần thưởng thưa thớt. Một trong những hệ quả tiêu cực và dai dẳng nhất của những giới hạn này chính là hiện tượng Hành vi lặp vòng (Looping Behavior).

Hành vi lặp vòng, hay còn được gọi là vòng lặp vô hạn (infinite loops) hoặc chu trình tuần hoàn (cyclic behavior), xảy ra khi tác tử bị mắc kẹt trong một quỹ đạo gồm các hành động lặp đi lặp lại mà không tạo ra bất kỳ sự tiến triển thực sự nào hướng tới mục tiêu cuối cùng của hệ thống. Hiện tượng này đặc biệt hiển hiện rõ nét trong các môi trường dạng lưới không gian (gridworlds) kinh điển như trò chơi Rắn săn mồi (Snake) hoặc bài toán điều hướng trong Mê cung (Maze navigation). 

Chẳng hạn, trong trò chơi Snake, khi chiều dài cơ thể của con rắn tăng lên, không gian trạng thái bùng nổ theo cấp số nhân, khiến việc học một chính sách hoàn hảo trở thành một bài toán thuộc lớp NP-hard. Khi đối mặt với rủi ro đâm vào chính cơ thể mình (nhận hình phạt lớn), mạng nơ-ron thường hội tụ về một chiến lược an toàn giả tạo: điều khiển con rắn di chuyển theo các vòng tròn hẹp hoặc dao động qua lại dọc theo các bức tường, từ chối việc tiếp cận thức ăn để bảo toàn mạng sống một cách vô nghĩa. Tương tự, trong các mê cung phức tạp, tác tử có thể di chuyển qua lại vô tận giữa hai điểm giao cắt vì hệ thống phần thưởng định hình sai lầm hoặc do không thể phân biệt được sự khác nhau giữa các khu vực do giới hạn về mặt biểu diễn trạng thái. 

Báo cáo nghiên cứu này thực hiện một cuộc khảo sát toàn diện và phân tích sâu sắc các nguyên nhân cấu trúc dẫn đến hành vi lặp vòng, đồng thời đánh giá chi tiết các giải pháp kiến trúc mạng nơ-ron tiên tiến nhất hiện nay để khắc phục triệt để vấn đề này.

---

## 1. Căn Nguyên Toán Học Của Quá Trình Quyết Định Markov Và Hành Vi Lặp Vòng

Để hiểu rõ bản chất của hành vi lặp vòng, trước tiên chúng ta phải phân tích hệ quy chiếu toán học nền tảng của học tăng cường: Quá trình Quyết định Markov (Markov Decision Process - MDP). Một MDP được định nghĩa một cách chặt chẽ bởi bộ ngũ phân $(S, A, P, R, \gamma)$, trong đó:
* $S$ đại diện cho không gian các trạng thái (state space)
* $A$ là không gian các hành động khả thi (action space)
* $P(s'|s,a)$ biểu thị xác suất chuyển đổi từ trạng thái $s$ sang trạng thái $s'$ khi thực hiện hành động $a$
* $R(s,a,s')$ là hàm phần thưởng cung cấp tín hiệu phản hồi ngay lập tức
* $\gamma \in [0, 1]$ là hệ số chiết khấu (discount factor) dùng để cân đối tầm quan trọng giữa phần thưởng hiện tại và phần thưởng tương lai.

Mục tiêu tối thượng của bất kỳ tác tử học tăng cường nào là tìm ra một chính sách tối ưu $\pi^*$, một quy tắc ánh xạ từ trạng thái sang hành động, sao cho giá trị kỳ vọng của tổng phần thưởng chiết khấu (expected cumulative discounted reward) đạt mức cực đại.

Dưới lăng kính của phương trình Bellman, hàm giá trị hành động tối ưu $Q^*(s, a)$ được kỳ vọng sẽ tính toán chính xác lợi ích dài hạn của việc thực hiện hành động $a$ tại trạng thái $s$, tạo ra một gradient dẫn đường rõ ràng giúp tác tử vượt qua mọi chướng ngại vật để đến đích. Tuy nhiên, giả định cốt lõi của MDP là "tính chất Markov" (Markov property): trạng thái hiện tại $s_t$ phải chứa đựng toàn bộ thông tin lịch sử cần thiết để đưa ra quyết định ở thời điểm $t$ mà không cần phải truy xuất lại các trạng thái trong quá khứ. Trong phần lớn các môi trường mô phỏng thực tế như Snake và Maze, giả định này bị phá vỡ một cách nghiêm trọng, biến bài toán thành một Quá trình Quyết định Markov Quan sát Một phần (Partially Observable Markov Decision Process - POMDP).

Trong môi trường POMDP, tác tử không được cấp quyền truy cập trực tiếp vào trạng thái thực sự của hệ thống $s_t$, mà chỉ nhận được một tín hiệu quan sát $o_t$ thông qua một hàm xác suất quan sát $O(o|s,a)$. Sự suy giảm thông tin này là nguồn cơn gốc rễ sinh ra hiện tượng lặp vòng. Khi tín hiệu quan sát không đủ phong phú để phản ánh cấu trúc toàn cục của môi trường, các thuật toán vốn được thiết kế cho MDP sẽ không thể gán giá trị tín dụng (credit assignment) một cách chính xác, dẫn đến việc thiết lập các giá trị Q-value bị sai lệch. Trong những điều kiện mơ hồ như vậy, tác tử có xu hướng lựa chọn các hành động bảo thủ tạo ra các vòng phản hồi cục bộ an toàn, thay vì thực hiện các chuỗi khám phá mạo hiểm dài hạn.

---

## 2. Hiện Tượng Trùng Lặp Trạng Thái (State Aliasing) Và Sự Sụp Đổ Của Không Gian Biểu Diễn

Hệ quả trực tiếp và nguy hiểm nhất của POMDP đối với các kiến trúc mạng nơ-ron là hiện tượng Trùng lặp Trạng thái (State Aliasing). Hiện tượng này xảy ra khi hai hoặc nhiều trạng thái vật lý độc lập trong môi trường tạo ra những tín hiệu quan sát giống hệt nhau (hoặc gần như không thể phân biệt được) đối với các cảm biến của tác tử. Khi thuật toán học tăng cường không thể phân tách các trạng thái này, mạng nơ-ron bị buộc phải ánh xạ chúng vào cùng một điểm hoặc một cụm vô cùng hẹp trong không gian biểu diễn ẩn (hidden representation space), dẫn đến sự sụp đổ về năng lực điều hướng.

McCallum (1996) đã cung cấp một minh chứng kinh điển cho sự thất bại này thông qua bài toán T-Maze đơn giản. Trong mô hình này, tác tử khởi hành từ một vị trí gốc $x_3$ hướng về phía Nam. Từ $x_3$, nó có hai lựa chọn: rẽ trái để tiến vào nhánh $x_2$, hoặc rẽ phải để tiến vào nhánh $x_1$. Sau khi tiến vào $x_1$ hoặc $x_2$, đường đi duy nhất tiếp theo là quay trở lại vị trí gốc $x_3$. Để đạt được quỹ đạo tối ưu, tác tử cần phải rẽ trái từ $x_3$ (hướng Nam) để vào $x_2$, sau đó khi đang ở $x_2$ (lúc này tác tử đang quay mặt về hướng Bắc), nó phải thực hiện hành động rẽ phải để quay về. 

Tuy nhiên, sự mù lòa về không gian xảy ra khi thiết kế quan sát của tác tử chỉ cho phép nó biết mình đang quay mặt về hướng Bắc hay Nam. Do cả hai trạng thái $x_1$ và $x_2$ đều có chung một đặc trưng quan sát là "quay mặt về hướng Bắc", tác tử rơi vào tình trạng State Aliasing. Mạng nơ-ron của tác tử không có cách nào biết được nó đang đứng ở nhánh trái hay nhánh phải.

Nghiên cứu trên các mạng nơ-ron hồi quy (RNN) được huấn luyện bằng các phương pháp tối ưu hóa chính sách (Policy Gradient) như REINFORCE đã cho thấy một kết quả đáng báo động khi đối mặt với T-Maze. Các thuật toán Policy Gradient không học trực tiếp giá trị của trạng thái, mà tối ưu hóa để xuất ra một phân phối xác suất hành động trực tiếp $\pi_\theta(a|s)$. Khi các trạng thái bị trùng lặp, ví dụ như $x_1$ và $x_2$, chúng buộc phải chia sẻ chung một phân phối hành động tối ưu nếu mô hình cập nhật dựa trên kỳ vọng gộp. Lực kéo của gradient (gradient descent) trong quá trình lan truyền ngược (backpropagation) ép khoảng cách Euclid giữa các biểu diễn ẩn của hai trạng thái này hội tụ về không ($||h_{x_1} - h_{x_2}||_2 \rightarrow 0$). 

Sự sụp đổ của không gian ẩn khiến mạng nơ-ron áp dụng một chính sách cứng nhắc, ví dụ như luôn rẽ phải khi nhìn thấy hướng Bắc. Hậu quả là, khi tác tử rơi vào trạng thái cần một hành động khác biệt, nó vẫn đưa ra quyết định rẽ phải, đưa nó trở lại vị trí cũ và tạo thành một vòng lặp vĩnh viễn (infinite loop) không thể bị phá vỡ nếu không có sự can thiệp từ bên ngoài.

---

## 3. Định Hình Phần Thưởng (Reward Shaping) Và Nguy Cơ Tạo Ra Điểm Cân Bằng Giả

Nguyên nhân thứ hai sinh ra hành vi lặp vòng, phổ biến đặc biệt trong môi trường Snake, bắt nguồn từ cấu trúc của hàm phần thưởng. Khi huấn luyện một tác tử điều khiển con rắn thông qua DRL, tín hiệu phần thưởng thưa thớt (chỉ có điểm khi ăn được thức ăn hoặc chết) làm chậm tốc độ hội tụ của thuật toán đến mức khó chấp nhận. Để giải quyết, các lập trình viên thường bổ sung các phần thưởng phụ trợ thông qua quá trình định hình phần thưởng (Reward Shaping). Chẳng hạn, họ cung cấp một phần thưởng nhỏ (+0.1) mỗi khi con rắn thu hẹp khoảng cách Manhattan hoặc khoảng cách Euclid tới quả táo, và một hình phạt nhỏ (-0.1) nếu nó di chuyển ra xa.

Tưởng chừng như logic này sẽ cung cấp một gradient tuyệt vời để dẫn đường, nhưng trong thực tế, các tác tử DRL, vốn là những cỗ máy tối ưu hóa tuyệt đối, sẽ nhanh chóng tìm ra các lỗ hổng của hàm heuristic này. Quá trình này được gọi là **Reward Hacking**. Nếu mạng nơ-ron học được rằng việc trườn quanh quả táo (tiến lại gần nhận +0.1, sau đó đi ngang không bị trừ điểm, rồi lại lùi ra nhận -0.1 nhưng ngay lập tức tiến vào nhận +0.1) có thể mang lại tổng phần thưởng dương vĩnh viễn, nó sẽ chọn cách thực hiện hành vi này vô tận thay vì đối mặt với rủi ro ăn quả táo (điều này sẽ làm con rắn dài ra và tăng khả năng đâm vào đuôi).

| Loại Phản Hồi | Cấu trúc Phần thưởng Thiết lập | Hành vi Không mong muốn của Tác tử DRL | Nguyên nhân Cơ học trong Mạng Nơ-ron |
| :--- | :--- | :--- | :--- |
| **Khoảng cách** | + $\epsilon$ nếu tiến gần mục tiêu, - $\epsilon$ nếu xa mục tiêu. | Dao động tiến/lùi hoặc xoay quanh mục tiêu. | Tác tử khai thác vòng lặp cục bộ để đạt điểm dương vĩnh viễn. |
| **Sống sót** | + $\epsilon$ cho mỗi bước đi không va chạm. | Di chuyển vòng tròn quanh biên hoặc cuộn tròn. | Trì hoãn việc hoàn thành màn chơi để tích lũy điểm sống sót. |
| **Phạt thời gian** | - $\epsilon$ cho mỗi bước di chuyển để ép thời gian. | Tự sát ngay từ bước đầu tiên để tránh bị trừ điểm. | Hàm giá trị (Value function) xác định mọi chiến lược đều âm nặng. |
| **Hình phạt vòng lặp** | - $X$ khi tọa độ đầu rắn trùng lặp qua thời gian. | Tìm các quỹ đạo phức tạp hơn, vòng vèo hơn. | Phụ thuộc vào kích thước bộ đệm (deque buffer) của hệ thống dò vết. |

Trong trò chơi Snake, các hiện tượng như vậy thường xảy ra khi điểm số (score) vượt ngưỡng trung bình (ví dụ khoảng 25-30 quả táo). Tại ngưỡng này, mạng nơ-ron định lượng rằng rủi ro tự va chạm đã quá lớn. Nó từ chối di chuyển vào tâm bản đồ để tiếp cận thức ăn, mà thay vào đó chạy lặp đi lặp lại quanh các bức tường ở cạnh bản đồ, hy vọng kéo dài sự sống mãi mãi (infinite survival loops). Lỗi logic này chứng minh rằng việc bổ sung các phần thưởng cục bộ ngây thơ (naive reward shaping) phá vỡ tính tối ưu tĩnh của bài toán gốc.

Để giải quyết vấn đề này ở mức độ toán học vững chắc, phương pháp Định hình Phần thưởng Dựa trên Thế năng (Potential-Based Reward Shaping - PBRS) đã được đề xuất bởi Ng, Harada và Russell (1999). PBRS không cấp phần thưởng tùy tiện, mà dựa trên sự khác biệt của một hàm thế năng $\Phi(s)$ được định nghĩa trên toàn bộ không gian trạng thái. Phần thưởng định hình $F$ đối với việc di chuyển từ trạng thái $s$ sang $s'$ thông qua hành động $a$ được tính bằng công thức:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

Cấu trúc này cực kỳ đặc biệt bởi tính chất triệt tiêu tuần hoàn (telescoping properties) của nó. Nếu một tác tử đi vào một vòng lặp kín trải qua các trạng thái $s_1 \rightarrow s_2 \rightarrow \dots \rightarrow s_n \rightarrow s_1$, tổng phần thưởng định hình nhận được qua toàn bộ vòng lặp (với giả định $\gamma \approx 1$) sẽ là:
$$F_{loop} = (\Phi(s_2) - \Phi(s_1)) + (\Phi(s_3) - \Phi(s_2)) + \dots + (\Phi(s_1) - \Phi(s_n)) = 0$$

Do tổng phần thưởng phụ trợ $F_{loop}$ dọc theo một vòng lặp luôn bằng 0 (hoặc nhỏ hơn 0 do chiết khấu $\gamma$), tác tử không thể khai thác bất kỳ phần thưởng giả tạo nào từ việc di chuyển vòng quanh. Sự bảo toàn tính tối ưu của chính sách (Policy Invariance) dưới phương pháp PBRS đảm bảo rằng không gian giá trị $Q(s,a)$ hội tụ về đúng mục tiêu cốt lõi mà không bị bóp méo bởi các điểm cực đại cục bộ.

---

## 4. Từ Mạng Feedforward Đến CNN: Sự Tiến Hóa Trong Biểu Diễn Môi Trường

Bên cạnh hàm phần thưởng, kiến trúc của chính mạng nơ-ron cũng quyết định năng lực của tác tử trong việc nhận diện và né tránh vòng lặp. Trong những dự án sơ khai áp dụng DQN cho Snake, trạng thái môi trường thường được tối giản hóa thành một véc-tơ một chiều để làm đầu vào cho Mạng nơ-ron Truyền thẳng (Feedforward Neural Networks - FNN). Một biểu diễn phổ biến nhất bao gồm 11 giá trị nhị phân (boolean): 3 cảm biến cảnh báo nguy hiểm (tường hoặc đuôi) ở ngay phía trước, bên trái, và bên phải đầu rắn; 4 biến one-hot encoding biểu diễn hướng di chuyển hiện tại; và 4 biến biểu diễn hướng tương đối của quả táo.

Sự hạn chế chết người của kiến trúc FNN sử dụng véc-tơ 11 chiều này là sự "mù lòa đường chéo" (diagonal blindness). Mạng nơ-ron hoàn toàn không thể nhận thức được các chướng ngại vật nằm ở khoảng cách lớn hơn 1 ô, và đặc biệt là không thể nhìn thấy các đoạn thân rắn nằm ở góc chéo. Hậu quả là, khi con rắn dài ra, nó thường tự cuộn cơ thể của mình thành một vòng lặp khép kín. Các cảm biến trực giao (trái, phải, trước) vẫn báo cáo là không gian an toàn (0), nhưng thực chất tác tử đã tự bao vây chính mình trong một không gian hình U hoặc hình O không có lối thoát. Khi lọt vào vòng lặp này, mạng nơ-ron hoàn toàn bất lực và chỉ đành chạy vòng quanh bên trong cho đến khi khoảng trống bị lấp đầy hoàn toàn.

Sự chuyển dịch sang Mạng Nơ-ron Tích chập (Convolutional Neural Networks - CNN) đã cung cấp một bước tiến mang tính cách mạng để giải quyết rào cản nhận thức không gian này. Thay vì nén mọi thứ vào một véc-tơ, không gian trò chơi được giữ nguyên hình thái hình học dưới dạng một ma trận đa kênh. Một kiến trúc chuẩn mực cho Snake sử dụng cấu trúc lưới không gian (ví dụ: lưới $10 \times 10$ hoặc $20 \times 20$) với nhiều lớp (channels) độc lập. Ví dụ, thiết lập 3 kênh ($3 \times H \times W$): kênh 1 chứa ma trận nhị phân đánh dấu vị trí các đốt thân rắn, kênh 2 đánh dấu vị trí thức ăn, và kênh 3 đánh dấu vị trí duy nhất của đầu rắn.

Thông qua các lớp tích chập (convolutional layers) trích xuất đặc trưng với các bộ lọc không gian (spatial kernels như $3 \times 3$ hoặc $5 \times 5$), CNN có thể dễ dàng nắm bắt các mô hình hình học toàn cục như cấu trúc vòng lặp đang hình thành, khoảng cách từ đầu đến tường, và các lối đi hẹp (bottlenecks). Khả năng cảm thụ không gian (spatial awareness) của CNN cho phép tác tử đưa ra các dự đoán giá trị $Q(s, a)$ dựa trên cấu trúc tổng thể, từ đó tự động né tránh việc tự nhốt mình vào các quỹ đạo đường hầm hoặc các vòng xoáy tử thần.

---

## 5. Kỹ Thuật Frame Stacking Và Giới Hạn Trong Trí Nhớ Ngắn Hạn

Mặc dù CNN cung cấp một tầm nhìn không gian xuất sắc, một khung hình (frame) tĩnh đơn lẻ vẫn không đủ để khôi phục tính chất Markov của môi trường. Trong một mê cung hoặc trên một lưới pixel, nhìn vào một khung hình tĩnh, mạng nơ-ron không thể biết đối tượng đang di chuyển về hướng nào và với vận tốc ra sao. Trong điều kiện thiếu thông tin động lượng này, tác tử có thể bị mắc kẹt vào các vòng lặp tiến-lùi (forward-backward loops) vì mạng nơ-ron đánh giá trạng thái hiện tại và trạng thái lùi lại một bước là hoàn toàn tương đương.

Giải pháp tiêu chuẩn công nghiệp được sử dụng trong các hệ thống DRL hiện đại (được DeepMind áp dụng thành công trên nền tảng Atari) là Kỹ thuật Chồng Khung hình (Frame Stacking). Thay vì đưa một khung hình duy nhất $o_t$ vào mạng CNN, hệ thống sẽ lưu trữ và nối (concatenate) $N$ khung hình gần nhất (thường là $N=4$) theo trục chiều sâu (channel axis) để tạo ra một tensor đầu vào duy nhất:
$$S_t = \{o_{t-3}, o_{t-2}, o_{t-1}, o_t\}$$

Sự kết hợp giữa Frame Stacking và các lớp tích chập không chỉ cung cấp định vị không gian mà còn cung cấp một luồng thời gian ngắn hạn, giúp mạng nơ-ron "cảm nhận" được gia tốc và chiều hướng vận động. Khi một tác tử lặp lại hành động trước đó, tensor quan sát của Frame Stacking giữa hai bước sẽ có sự khác biệt rõ rệt về chuỗi thời gian, phá vỡ ảo giác về sự tương đương trạng thái (State Aliasing) trong phạm vi $N$ bước. 

Frame Stacking vô cùng hiệu quả trong việc loại bỏ các rung lắc nhỏ (micro-oscillations) và các vòng lặp chu kỳ ngắn. Tuy nhiên, nó lại thất bại hoàn toàn trước các vòng lặp có độ dài vượt quá giới hạn của cửa sổ xếp chồng. Nếu tác tử tạo ra một vòng lặp dài 10 bước trong khi $N=4$, hệ thống Frame Stacking sẽ lại đánh mất bối cảnh lịch sử, và con rắn sẽ tiếp tục xoay vòng vĩnh viễn.

---

## 6. Deep Recurrent Q-Networks (DRQN) Khôi Phục Dòng Thời Gian Bị Mất

Để khắc phục ranh giới hạn hẹp của bộ nhớ Frame Stacking, Hausknecht và Stone (2015) đã giới thiệu một biến thể cấu trúc được gọi là Deep Recurrent Q-Networks (DRQN). Kiến trúc này tích hợp cơ chế của Mạng Nơ-ron Hồi quy (Recurrent Neural Networks), cụ thể là các khối Bộ nhớ Ngắn-Dài hạn (Long Short-Term Memory - LSTM), trực tiếp vào đường ống xử lý của thuật toán Q-Learning.

Kiến trúc chuẩn của DRQN bao gồm một mạng CNN ở các lớp đầu vào để trích xuất các đặc trưng không gian đa chiều từ hình ảnh pixel thuần túy, sau đó, thay vì sử dụng các lớp kết nối đầy đủ (fully connected layers) phẳng như thông thường, thông tin được đưa vào một lớp LSTM. Cấu trúc toán học của lớp LSTM cho phép duy trì và cập nhật một trạng thái ẩn (hidden state) $h_t$ và trạng thái tế bào (cell state) $c_t$ qua từng bước thời gian:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Cổng Quên)}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Cổng Đầu Vào)}$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \ast c_{t-1} + i_t \ast \tilde{c}_t \quad \text{(Trạng thái Tế bào Mới)}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Cổng Đầu Ra)}$$
$$h_t = o_t \ast \tanh(c_t)$$

Đầu ra $h_t$ sau đó được truyền qua một lớp tuyến tính cuối cùng để ước tính các giá trị Q cho tất cả các hành động khả thi.

Sức mạnh thực sự của DRQN trong việc đối phó với hiện tượng Hành vi Lặp vòng (Looping Behavior) nằm ở khả năng lan truyền bối cảnh lịch sử (historical context) xuyên suốt một khoảng thời gian dài bất định. Khi một tác tử DRQN trong môi trường Maze di chuyển qua một dãy hành lang dài và gặp một ngã tư trông giống hệt như một ngã tư nó đã đi qua 20 bước trước đó, không gian quan sát (observation) ở cả hai thời điểm có thể trùng khớp hoàn toàn, nhưng véc-tơ trạng thái ẩn $h_t$ thì không. Lớp LSTM đã ghi nhớ hành trình trước đó thông qua cổng tế bào $c_t$, do đó nó tạo ra những đầu ra Q-value phân kỳ cho cùng một đầu vào thị giác. Nhờ đó, DRQN tự động bẻ gãy sự ràng buộc của State Aliasing, giúp tác tử tránh việc lặp lại quyết định sai lầm trong quá khứ và khám phá các lối thoát mới. 

Tuy vậy, DRQN đòi hỏi kỹ thuật huấn luyện cực kỳ phức tạp thông qua quá trình Truyền ngược qua thời gian (Backpropagation Through Time - BPTT) và đòi hỏi việc thiết kế bộ đệm Replay Memory phải lưu trữ các quỹ đạo tuần tự thay vì các bước độc lập, làm tăng đáng kể chi phí tính toán.

---

## 7. Trí Nhớ Sự Kiện (Episodic Memory) Và Kiến Trúc Transformer Trong Học In-Context

Trong khi DRQN đại diện cho một bước nhảy vọt so với CNN, sự giới hạn về dung lượng bộ nhớ của véc-tơ ẩn $h_t$ khiến kiến trúc này gặp khó khăn trong những mê cung quy mô lớn (large-scale mazes), nơi các vòng lặp có thể trải dài qua hàng nghìn bước. Sinh học thần kinh cung cấp một giải pháp tốt hơn: động vật có vú sử dụng vùng hải mã (hippocampus) và cấu trúc trí nhớ sự kiện (episodic memory) để lưu trữ và truy xuất lại những kinh nghiệm cụ thể trong quá khứ để tránh các cạm bẫy vòng lặp. Những nghiên cứu sinh học ghi nhận hoạt động của các tế bào vị trí (place cells) và tế bào thời gian (time cells) trong phân vùng CA1 và CA3 của hải mã đã định hình nên các mô hình Trí nhớ Sự kiện kiểm soát tuần tự (Sequential Episodic Control - SEC).

Trong lĩnh vực trí tuệ nhân tạo đương đại, kiến trúc Transformer đã trở thành đối trọng hoàn hảo cho hệ thống đồi hải mã này. Cơ chế Attention (Cơ chế chú ý) dựa trên Khóa - Giá trị (Key-Value) trong Transformer phản ánh trực tiếp cơ chế lập chỉ mục (indexing mechanism) của trí nhớ sự kiện. Thay vì cố gắng nén toàn bộ thông tin quá khứ vào một trạng thái $h_t$ duy nhất có kích thước cố định như LSTM, kiến trúc Transformer bảo lưu toàn bộ quỹ đạo của tác tử dưới dạng một chuỗi các Token độc lập lưu trữ trong cửa sổ ngữ cảnh (context window):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Khi một tác tử tích hợp Transformer điều hướng trong mê cung, mỗi khung cảnh (scene) hoặc trạng thái được chuyển đổi thành các biểu diễn Query (truy vấn), Key (khóa), và Value (giá trị). Nếu tác tử đi vào một quỹ đạo lặp vòng và đối diện với một ngã rẽ mà nó từng gặp phải, véc-tơ Query của nó sẽ thực hiện phép tính độ tương đồng góc (dot-product similarity) với toàn bộ các véc-tơ Key trong quá khứ. Nó sẽ ngay lập tức "bắt sóng" (high attention weight) với cảnh quay giống hệt trong quá khứ, sau đó truy xuất lại véc-tơ Value (nội dung trí nhớ, chẳng hạn như hành động trước đó đã dẫn đến đi vào ngõ cụt).

Hơn thế nữa, Transformer có khả năng phi thường trong việc thực hiện Học tăng cường Trong-ngữ cảnh (In-context Reinforcement Learning). Bằng cách caching (lưu trữ tạm thời) các tính toán giá trị trung gian trong các token bộ nhớ, mô hình Transformer có thể sử dụng trí nhớ sự kiện như một không gian làm việc thuật toán (computational workspace). Sự linh hoạt này giúp nó tự động đảo ngược các chiến lược thất bại hoặc tái thiết lập quỹ đạo để bẻ gãy vòng lặp mà không cần phải thực hiện bất kỳ sự thay đổi nào đối với trọng số của mạng (no weight updates required during inference). Điều này mở ra kỷ nguyên mới của các tác tử Zero-shot có khả năng thích ứng siêu tốc trong các mê cung có kiến trúc xa lạ (tree mazes, gridworlds).

---

## 8. Động Lực Nội Tại (Intrinsic Motivation) Và Cơ Chế Khám Phá Sự Tò Mò (Curiosity)

Bên cạnh giải pháp cấu trúc về trí nhớ (Memory), một hướng đi đột phá khác nhằm khắc phục hành vi lặp vòng trong các không gian phần thưởng thưa thớt (sparse reward domains) là tái cấu trúc lại động lực của tác tử, mô phỏng quá trình khám phá của sinh vật thông qua Động lực Nội tại (Intrinsic Motivation) hay còn gọi là Sự Tò mò (Curiosity).

Trong các mê cung rộng lớn, nơi mà tín hiệu phần thưởng ngoại sinh (extrinsic reward) bằng không cho đến khi tác tử giải quyết được bài toán, các thuật toán thăm dò ngẫu nhiên như $\epsilon$-greedy là hoàn toàn vô dụng, khiến tác tử sa vào vô vàn các vòng lặp di chuyển ngẫu nhiên và quẩn quanh tại một chỗ. Các phương pháp Khám phá Dựa trên Sự Tò mò (Curiosity-driven Exploration) giải quyết sự bế tắc này bằng cách cung cấp cho tác tử các phần thưởng nội tại (intrinsic rewards) cho việc khám phá các khu vực mới.

Cơ chế trung tâm của mô hình này bao gồm một Mô hình Động lực học Thuận (Forward Dynamics Model) được huấn luyện song song với tác tử. Mô hình này (có thể là một mạng đa tầng MLP hoặc CNN) nhận đầu vào là biểu diễn trạng thái ở thời điểm $t$ ($s_t$) và hành động được chọn $a_t$, để dự đoán biểu diễn trạng thái tiếp theo ở thời điểm $t+1$ ($\hat{s}_{t+1}$). Tín hiệu phần thưởng nội tại $R_{intrinsic}$ được tính bằng độ lớn của sai số dự đoán (prediction error):
$$R_{intrinsic} = ||\hat{s}_{t+1} - s_{t+1}||^2_2$$

Hoặc, theo cách tiếp cận lý thuyết thông tin như Sự Ngạc nhiên Bayes (Bayesian surprise), nó được đo lường bằng phân kỳ Kullback-Leibler giữa phân phối hậu nghiệm và tiên nghiệm của mô hình tiềm ẩn (latent variable):
$$r_{intrinsic}(S) = D_{KL}(p(Z\|S) \| p(Z))$$

### Cơ chế phá vỡ vòng lặp:
Sự tò mò hoạt động như một khắc tinh tự nhiên của hành vi lặp vòng. Nếu một tác tử đi vào một vòng lặp kín, các quan sát và trạng thái trong quỹ đạo đó sẽ liên tục lặp lại. Mô hình Forward Dynamics sẽ rất nhanh chóng học được các quy luật của vòng lặp này và bắt đầu đưa ra các dự đoán hoàn toàn chính xác về trạng thái tiếp theo. 

Khi sự dự đoán trở nên hoàn hảo, sai số giảm xuống bằng $0$, dẫn đến việc phần thưởng nội tại bị triệt tiêu hoàn toàn ($R_{intrinsic} \rightarrow 0$). Sự "nhàm chán" (boredom) này làm suy giảm giá trị $Q(s, a)$ của các hành động thuộc vòng lặp, tự động tạo ra một áp lực gradient ép tác tử phải đi tìm một khu vực chưa từng được khám phá, nơi sai số dự đoán còn cao, để "thỏa mãn" sự tò mò của nó. Thông qua đó, mô hình Tò mò tự động dập tắt mọi hành vi lặp vòng tiêu cực, thúc đẩy tác tử lập bản đồ toàn bộ môi trường và hoàn thiện mục tiêu tối thượng mà không cần sự định hình phần thưởng thủ công từ các kỹ sư.

---

## 9. Sự Nhầm Lẫn Chính Sách (Policy Confounding) Và Rào Cản Tổng Quát Hóa Ngoài Quỹ Đạo (Out-of-Trajectory Generalization)

Một khía cạnh tinh vi và chỉ mới được lý giải gần đây trong các thất bại của học tăng cường (đặc biệt liên quan đến hành vi lặp vòng) là khái niệm Sự Nhầm lẫn Chính sách (Policy Confounding) và sự mất năng lực Tổng quát hóa Ngoài Quỹ đạo (Out-of-Trajectory - OOT).

Trong giai đoạn đầu của RL, tác tử khám phá môi trường một cách rộng rãi. Tuy nhiên, khi chính sách bắt đầu hội tụ, tác tử thu hẹp các thao tác của mình vào một tập hợp các quỹ đạo (trajectories) nhất định. Quá trình chọn lọc này, một cách không vô ý, đã tạo ra sự thiên lệch lớn trong tập dữ liệu huấn luyện, giới hạn không gian tương tác và sinh ra các Tương quan Giả (Spurious Correlations) giữa các quan sát từ môi trường và phần thưởng.

Mạng nơ-ron là các cỗ máy nhận diện mẫu (pattern recognition) cực đoan, chúng sẽ nắm bắt bất kỳ đặc trưng nào (cho dù là vô nghĩa về mặt nguyên nhân - kết quả) miễn là đặc trưng đó giúp dự đoán chính xác giá trị $Q$. Ví dụ, trong một mê cung, nếu đường đi tối ưu nhất đến mục tiêu luôn đi ngang qua một ô gạch màu đỏ, mạng nơ-ron sẽ hình thành một "thói quen" (habit): ánh xạ trực tiếp sự xuất hiện của ô gạch màu đỏ với hành động "rẽ trái". Thói quen này (Bad Habit) hoạt động hoàn hảo miễn là tác tử còn đi trên quỹ đạo tối ưu (in-trajectory).

Sự cố chí mạng xảy ra khi môi trường bị xáo thiện nhẹ (như gió thổi lệch hướng, hoặc vị trí xuất phát thay đổi ngẫu nhiên). Tác tử bị văng ra khỏi quỹ đạo quen thuộc của nó (Out-of-Trajectory). Tại một khu vực xa lạ trong mê cung, nó tình cờ bắt gặp một ô gạch màu đỏ khác nằm ở một ngã cụt. Kích hoạt thói quen cũ, tác tử "rẽ trái", đâm sầm vào tường. Nó nảy ra, lại nhìn thấy ô gạch đỏ, lại thực hiện hành động "rẽ trái". Quá trình này hình thành một vòng lặp vĩnh viễn. Mạng nơ-ron không thể thoát ra do nó không hiểu được quy luật cấu trúc nhân quả (causality) của mê cung, mà chỉ lưu giữ những ký ức tương quan giả định do chính quỹ đạo cũ của nó sinh ra.

Các nhà nghiên cứu chỉ ra rằng các thuật toán cập nhật trên-chính-sách (On-policy algorithms) như PPO đặc biệt dễ bị tổn thương trước hiện tượng Policy Confounding hơn các phương pháp ngoài-chính-sách (Off-policy) như DQN. Trong DQN, việc sử dụng bộ nhớ Replay Buffer cho phép trộn lẫn các mẫu từ các chính sách cũ (khám phá ngẫu nhiên) với chính sách mới, giúp duy trì sự đa dạng của dữ liệu (data diversity) và loại bỏ dần các tương quan giả. Hơn nữa, việc thêm các lớp điều chuẩn (regularization) vào lớp biểu diễn ẩn hoặc duy trì tính ngẫu nhiên thông qua Epsilon-Greedy hoặc Entropy Bonus là các phương pháp cốt lõi để duy trì sự đa dạng và bẻ gãy các thói quen vòng lặp độc hại.

---

## 10. Các Công Cụ Kỹ Thuật Phụ Trợ Trong Điều Chỉnh Hệ Thống

Ngoài các kiến trúc lý thuyết chuyên sâu, trong thực tiễn kỹ thuật để giải quyết dứt điểm hành vi lặp vòng trong các hệ thống RL như Snake, các nhóm phát triển thường thiết lập một mạng lưới các công cụ bảo vệ (safeguards) và tối ưu hóa hệ thống (optimizations):

1. **Bộ Phát Hiện Chu Trình Hành Động (Action Cycle Detector) và Trình Theo Dõi Quỹ Đạo (Path Tracer)**: Thay vì chờ đợi mạng nơ-ron tự sửa sai, các mô-đun phần mềm (logic rules) được đính kèm vào môi trường để theo dõi lịch sử tọa độ của tác tử (như thuật toán dò vết - path tracer, sử dụng deque queue). Nếu tác tử đi qua cùng một điểm không gian nhiều lần mà không đạt được trạng thái mới (không ăn thêm điểm), hệ thống chủ động cung cấp một phần thưởng âm lũy tiến (progressive negative penalty, ví dụ: lặp 1 vòng phạt -10, lặp 2 vòng phạt -20) để hủy diệt hàm Q-value cục bộ của quỹ đạo đó, buộc mạng nơ-ron phải chọn một hướng khác.
2. **Double DQN (DDQN) và Dueling DQN**: Việc sử dụng một mạng Q-Network duy nhất thường dẫn đến hiện tượng Ước tính quá mức (Overestimation Bias), nơi giá trị Q của các hành động lặp vòng được khuếch đại một cách phi thực tế. Thuật toán Double DQN tách rời quá trình chọn hành động (bởi mạng trực tuyến - online network) và đánh giá hành động (bởi mạng mục tiêu - target network), làm phẳng các dao động giá trị và giữ ổn định chính sách. Dueling DQN tiếp tục chia tách mạng thành hai luồng để tính toán Giá trị Trạng thái (State Value) và Lợi thế Hành động (Action Advantage), cho phép tác tử nhận diện các trạng thái "chết" (dead states - nới vòng lặp xảy ra) nhanh hơn mà không cần thử mọi hành động.
3. **Các Chiến Lược Thuật Toán Trọng Tài (Heuristic Fallbacks)**: Khi DRL thất bại và rơi vào vòng lặp (thường gặp khi rắn đã phủ kín 80% bản đồ), hệ thống được cấu hình để chuyển giao quyền kiểm soát cho các thuật toán truyền thống dựa trên cây tìm kiếm, chẳng hạn như Thuật toán Tìm đường A*, Thuật toán Tìm kiếm Chiều rộng (BFS), hoặc duy trì một Chu trình Hamilton (Hamiltonian Cycle) thông qua thuật toán Prim. Sự lai ghép này (Hybrid AI) bảo đảm hiệu suất trần cao trong khi chờ mạng nơ-ron cập nhật trọng số.

---

## Kết Luận

Nghiên cứu toàn diện về hành vi lặp vòng (Looping Behavior) trong Học Tăng Cường, trải dài từ các miền không gian nhỏ gọn như Snake đến các mê cung phức tạp (Mazes), tiết lộ rằng sự thất bại của các tác tử không phải do sự khiếm khuyết của toán học tối ưu, mà do những giới hạn cấu trúc về biểu diễn và sự suy biến của luồng thông tin.

Những phân tích từ bài toán T-Maze của McCallum đến hiện tượng nhầm lẫn chính sách (Policy Confounding) đã minh chứng rằng: để một trí tuệ nhân tạo có thể thoát khỏi chiếc bóng lặp đi lặp lại của chính nó, hệ thống phải được trang bị năng lực hồi tưởng thời gian. Việc nâng cấp từ các mạng truyền thẳng (Feedforward) một chiều sang Mạng Nơ-ron Tích chập (CNN) kết hợp Frame Stacking chỉ giải quyết được phần ngọn của vấn đề. Các giải pháp gốc rễ yêu cầu việc áp dụng Mạng Nơ-ron Hồi quy Sâu (DRQN với LSTM) để khôi phục tính Markov bị ẩn, hoặc tiến xa hơn là mô phỏng sinh học bằng cấu trúc Trí nhớ Sự kiện (Episodic Memory) dựa trên cơ chế Attention của kiến trúc Transformer.

Hơn nữa, thiết kế của hệ thống phần thưởng đóng vai trò định hướng sự sống còn. Sự sụp đổ của các tác tử trước sức hút của các vòng lặp điểm số (reward hacking) chỉ ra ranh giới nguy hiểm của kỹ thuật định hình phần thưởng thủ công. Các hệ thống phần thưởng cần phải dựa trên các nguyên tắc toán học vững chắc như Định hình Phần thưởng Dựa trên Thế năng (PBRS) nhằm đảm bảo sự bất biến của chính sách tối ưu. 

Cuối cùng, việc tích hợp động lực nội tại (Intrinsic Motivation/Curiosity) để thúc đẩy tác tử liên tục khám phá các ranh giới mới của môi trường thông qua sai số dự đoán (prediction errors) không chỉ đập tan mọi hành vi lặp vòng nhàm chán, mà còn đặt nền móng cho các hệ thống Học Tăng Cường có khả năng tự động hóa và tổng quát hóa cao độ trong tương lai. Những đổi mới liên tục trong kiến trúc Neural Network và hệ thống thuật toán này đang dần xóa nhòa ranh giới giữa một đoạn mã cơ học và một thực thể AI với khả năng nhận thức toàn vẹn không gian và thời gian.
